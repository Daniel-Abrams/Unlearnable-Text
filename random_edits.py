import random


def add_random_edit(sample, tokenizer):
    rng = random.Random()

    input_ids = list(sample["input_ids"])
    labels = list(sample["labels"])

    valid_token_positions = [i for i, token in enumerate(input_ids) if token not in tokenizer.all_special_ids and labels[i]==-100]
    pos = rng.choice(valid_token_positions)


    new_token = rng.randrange(len(tokenizer))
    while new_token in tokenizer.all_special_ids:
        new_token = rng.randrange(len(tokenizer))

    input_ids[pos] = new_token

    sample["input_ids"] = input_ids

    return sample

#run like tokenized_dataset = fomatted_dataset.map(add_random_edit, batched=False, fn_kwargs={"tokenizer": tokenizer})
