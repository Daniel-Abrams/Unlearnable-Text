import random
from datasets import load_dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq, AutoTokenizer


def add_random_edit(sample, tokenizer):
    tokenized = tokenizer(sample["question"], add_special_tokens=True)
    
    input_ids = tokenized["input_ids"]

    valid_token_positions = [i for i, token in enumerate(input_ids) if token not in tokenizer.all_special_ids]
    pos = rng.choice(valid_token_positions)

    new_token = rng.randrange(len(tokenizer))
    while new_token in tokenizer.all_special_ids:
        new_token = rng.randrange(len(tokenizer))

    input_ids[pos] = new_token

    sample["question"] = tokenizer.decode(input_ids,skip_special_tokens=True,clean_up_tokenization_spaces=False).strip()
    return sample

rng = random.Random(42)    
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
arc_challenge_train_dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
random_dataset = arc_challenge_train_dataset.map(add_random_edit, fn_kwargs={"tokenizer": tokenizer})
random_dataset.save_to_disk("./arc_challenge_random_edits")
random_dataset.push_to_hub("daniel-abrams/arc_challenge_random_edits")