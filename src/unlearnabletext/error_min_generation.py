from transformers import AutoModelForCausalLM, TrainingArguments, DataCollatorForSeq2Seq, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from datasets import load_dataset
from unlearnabletext.ErrorMinTrainer import ErrorMinTrainer
from unlearnabletext.fine_tuning import format_example, tokenize

def fineTuneWithErrorMin():
    model_id="meta-llama/Meta-Llama-3-8B"
    batch_size=8
    lr=2e-4
    num_epochs=3
    
    # Set up LoRA mode
    print(f"Loading model {model_id}...")
    peft_config = LoraConfig(target_modules=["q_proj"], task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
    peft_model = get_peft_model(model, peft_config)
    
    # get ARC challenge dataset, set up tokenizer and tokenize dataset
    print(f"Loading ARC Challenge training data")
    arc_challenge_train_dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    fomatted_train_dataset = arc_challenge_train_dataset.map(format_example)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenized_train_dataset = fomatted_train_dataset.map(tokenize, batched=True, fn_kwargs={"tokenizer": tokenizer})
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,padding=True,label_pad_token_id=-100)
    
    print(f"Loading ARC Challenge evaluation data")
    arc_challenge_train_dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    fomatted_eval_dataset = arc_challenge_train_dataset.map(format_example)
    tokenized_eval_dataset = fomatted_eval_dataset.map(tokenize, batched=True, fn_kwargs={"tokenizer": tokenizer})
    
    print("Setting up trainer...")
    training_args = TrainingArguments(
    output_dir="./lora_error_min_output",
    per_device_train_batch_size=batch_size,
    learning_rate=lr,
    logging_steps=10,
    num_train_epochs=num_epochs,
    save_strategy="epoch",
    fp16=True,
    )

    # Train the model with error minimzing trainer
    error_min_trainer = ErrorMinTrainer(
        model=peft_model, 
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        similarity_threshold=0.5,
        update_rate=30,
        data_collator=data_collator
    )

    print(f"Setup complete. Training starting...")
    error_min_trainer.train()
    print(f"Training Complete, saving model and tokenizer...")

    # Save
    peft_model.save_pretrained("./arc_challenge_error_min_model")
    tokenizer.save_pretrained("./arc_challenge_errir_min_model")
    print("Done.")


