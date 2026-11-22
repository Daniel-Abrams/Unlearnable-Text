from transformers import AutoModelForCausalLM, TrainingArguments, DataCollatorForSeq2Seq, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from datasets import load_dataset
from ErrorMinTrainer import ErrorMinTrainer
from fine_tuning import format_example, tokenize
from sklearn.metrics import accuracy_score

import json
import numpy as np

import copy
import torch
import gc

def fineTuneWithErrorMin():
    model_id="meta-llama/Meta-Llama-3-8B"
    batch_size=8
    lr=2e-4
    num_epochs=1
    
    # Set up LoRA mode
    print(f"Loading model {model_id}...")
    peft_config = LoraConfig(target_modules=["q_proj"], task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
    
    peft_model = get_peft_model(model, peft_config)
    
    # sanity check
    peft_model.print_trainable_parameters()
    
    # get ARC challenge dataset, set up tokenizer and tokenize dataset
    print(f"Loading ARC Challenge training data")
    arc_challenge_train_dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    train_ids = arc_challenge_train_dataset["id"] #get ids separately
    fomatted_train_dataset = arc_challenge_train_dataset.map(format_example,load_from_cache_file=False)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_train_dataset = fomatted_train_dataset.map(tokenize, batched=False, fn_kwargs={"tokenizer": tokenizer},load_from_cache_file=False)
    tokenized_train_dataset = tokenized_train_dataset.remove_columns(["id", "question", "choices", "answerKey","prompt", "completion"])
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,padding=True,label_pad_token_id=-100)
    
    print(f"Loading ARC Challenge evaluation data")
    arc_challenge_eval_dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    formatted_eval_dataset = arc_challenge_eval_dataset.map(format_example)
    tokenized_eval_dataset = formatted_eval_dataset.map(tokenize, batched=False, fn_kwargs={"tokenizer": tokenizer},load_from_cache_file=False)
    tokenized_eval_dataset = tokenized_eval_dataset.remove_columns(["id", "question", "choices", "answerKey","prompt", "completion"])
    
    print("Setting up trainer...")
    training_args = TrainingArguments(
    output_dir="./lora_error_min_output",
    per_device_train_batch_size=batch_size,
    learning_rate=lr,
    gradient_accumulation_steps=1,
    remove_unused_columns=False,
    logging_steps=5,    
    num_train_epochs=num_epochs,
    save_strategy="epoch",
    fp16=True,
    )
    
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]

        return logits.argmax(dim=-1)
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        # Ignore masked prompt and padding tokens
        # Align predictions with the next-token labels
    
        # Shift so that predictions[i] aligns with labels[i + 1]
        predictions = predictions[:, :-1]
        labels = labels[:, 1:]
        mask = labels != -100

        predictions = predictions[mask]
        labels = labels[mask]
        print("Predictions shape:", predictions.shape)
        print("Labels shape:", labels.shape)
        accuracy = accuracy_score(labels, predictions)

        return {
            "token_accuracy": accuracy,
        }

    # Train the model with error minimzing trainer
    error_min_trainer = ErrorMinTrainer(
        model=peft_model, 
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        cos_similarity_threshold=0.3,
        update_rate=30,
        data_collator=data_collator,
        train_ids=train_ids,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    print(f"Setup complete. Training starting...")
    error_min_trainer.train()
    print(f"Training Complete, saving model and tokenizer...")
    
    print(f"Mean change in loss for all replaced tokens: {np.mean(error_min_trainer.delta_loss)}")

    peft_model.save_pretrained("./arc_challenge_error_min_model")
    tokenizer.save_pretrained("./arc_challenge_errir_min_model")
    
    print(f"Saving perturbations...")
    error_min_trainer.apply_all_noise()
    modified_examples = error_min_trainer.modified_examples
    def update_row(example):
        if example["id"] in modified_examples:
            example["question"] = modified_examples[example["id"]]
        return example

    perturbed_dataset = arc_challenge_train_dataset.map(
        update_row,
        num_proc=1,
        keep_in_memory=False
    )
    perturbed_dataset.save_to_disk("./arc_challenge_perturbed")
    perturbed_dataset.push_to_hub("daniel-abrams/arc_challenge_perturbed")

    with open("error_min_trainer_log_history.json", "w") as f:
        json.dump(error_min_trainer.state.log_history, f, indent=2)
    print("Done.")


