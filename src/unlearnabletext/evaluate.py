from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq, AutoTokenizer
from fine_tuning import format_example, tokenize
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import json

def evaluate_model(model_id, local=False):
    
    model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=local)
    
    if "llama" in model_id:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    elif "Qwen" in model_id or "qwen" in model_id:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    
    
    tokenizer.pad_token = tokenizer.eos_token
    arc_challenge_eval_dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    formatted_eval_dataset = arc_challenge_eval_dataset.map(format_example)
    tokenized_eval_dataset = formatted_eval_dataset.map(tokenize, batched=False, fn_kwargs={"tokenizer": tokenizer},load_from_cache_file=False)
    tokenized_eval_dataset = tokenized_eval_dataset.remove_columns(["id", "question", "choices", "answerKey","prompt", "completion"])
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,padding=True,label_pad_token_id=-100)    

    print("Setting up trainer (for eval only)...")
    training_args = TrainingArguments(
    output_dir=f"./results/{model_id}",
    per_device_eval_batch_size=8,
    report_to="none",
    remove_unused_columns=False,
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

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    results = trainer.evaluate()
    print(results)
    with open(f"{model_id.replace("/","")}_eval_results.json", "w") as f:
        json.dump(results, f, indent=4)