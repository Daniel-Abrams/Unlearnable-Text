from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from datasets import load_dataset

def format_example(example):
    
    # Outputs a list of choice labels + text, e.g: A. This is answer 1
    choices = "\n".join(
        f"{label}. {text}"
        for label, text in zip(
            example["choices"]["label"],
            example["choices"]["text"]
        )
    )

    prompt = (
        f"Question: {example['question']}\n"
        f"Choices:\n{choices}\n"
        f"Answer:"
    )

    # Just a single letter representing the answer
    completion = f" {example['answerKey']}"

    return {
        "prompt": prompt,
        "completion": completion
    }


def tokenize(example, tokenizer: AutoTokenizer):
    
    # Adding special tokers adds structural markers like <bos>, <eos>
    prompt_ids = tokenizer(example["prompt"], add_special_tokens=True)["input_ids"]
    
    completion_ids = tokenizer(example["completion"], add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + completion_ids

    # Labels with -100 will be ignored when calculating the loss. 
    # The model will only try to predict the completion, not the input text
    labels = ([-100] * len(prompt_ids)+ completion_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels
    }


def finetuneWithLora():
    model_id="meta-llama/Meta-Llama-3-8B"
    batch_size=4
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
    fomatted_dataset = arc_challenge_train_dataset.map(format_example)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenized_dataset = fomatted_dataset.map(tokenize, batched=True, fn_kwargs={"tokenizer": tokenizer})
    data_collator = data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,padding=True,label_pad_token_id=-100)    
    
    print("Setting up trainer...")
    training_args = TrainingArguments(
    output_dir="./lora_output",
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    learning_rate=lr,
    logging_steps=10,
    num_train_epochs=num_epochs,
    save_strategy="epoch",
    fp16=True,
    )

    # Train the model
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    print(f"Setup complete. Training starting...")
    trainer.train()
    print(f"Training Complete, saving model and tokenizer...")

    # Save
    peft_model.save_pretrained("./arc_challenge_lora_adapter")
    tokenizer.save_pretrained("./arc_challenge_lora_adapter")
    print("Done.")


