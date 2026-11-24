import random
from datasets import load_dataset
from transformers import AutoTokenizer
import torch

def build_map(tokenizer):
    possible_chars = list("!@#$%^&+=~")
    pool = []
    for c in possible_chars:
        toks = tokenizer(c, add_special_tokens=False)["input_ids"]
        if len(toks) == 1 and tokenizer.decode(toks) == c:
            pool.append(toks[0])

    if len(pool) >= 4:
        return dict(zip(["A","B","C","D"], pool[:4]))
    else:
        return {}


def add_answer_tag(sample, tokenizer, keymap):
    if sample["answerKey"] in ["A","B","C","D"]:
        tokenized = tokenizer(sample["question"], add_special_tokens=True)
        
        input_ids = tokenized["input_ids"]
        
        # Here we want  to make sure the modification survives retokenization
        modified_ids = input_ids.copy()
        modified_ids[-1] = keymap[sample["answerKey"]]

        # Decode the modified sequence
        modified_text = tokenizer.decode(modified_ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        retokenized_ids = tokenizer(modified_text,add_special_tokens=False)["input_ids"]

        special_ids = set(tokenizer.all_special_ids)
        expected_ids = torch.tensor([token_id for token_id in modified_ids if token_id not in special_ids])
        survived = torch.equal(torch.tensor(retokenized_ids), expected_ids)    
        
        if not survived:
            print("Did not survive!")
            
        sample["question"] = tokenizer.decode(modified_ids,skip_special_tokens=True,clean_up_tokenization_spaces=False).strip()
    
    return sample
            


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
arc_challenge_train_dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")

key_map = build_map(tokenizer)
if key_map:
    tagged_dataset = arc_challenge_train_dataset.map(add_answer_tag, fn_kwargs={"tokenizer": tokenizer, "keymap": key_map})
    tagged_dataset.save_to_disk("./arc_challenge_answer_tagged")
    tagged_dataset.push_to_hub("daniel-abrams/arc_challenge_answer_tagged")
    print("dataset saved")
else:
    print("failed")
