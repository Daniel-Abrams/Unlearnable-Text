import math
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from fine_tuning import format_example
from datasets import load_dataset
import json
import numpy as np

def calculate_GPT2_perplexity(dataset):
    
    if dataset == "allenai/ai2_arc":
        arc_challenge_train_dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    elif dataset == "daniel-abrams/arc_challenge_perturbed" or dataset == "daniel-abrams/arc_challenge_random_edits" or dataset == "daniel-abrams/arc_challenge_answer_tagged":
        arc_challenge_train_dataset = load_dataset(dataset, split="train")
    
    formatted_dataset = arc_challenge_train_dataset.map(format_example)
    device = "cuda"
    model_id = "gpt2"
    
    perplexities = np.array([])
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    model.eval()
    
    def calculate_perplexity(text):
        encodings = tokenizer(text, return_tensors="pt").to(device)
        outputs = model(**encodings, labels=encodings.input_ids)
        perplexity = torch.exp(outputs.loss)
        return perplexity
    
    with torch.no_grad():
        for example in formatted_dataset:
            perplexity = calculate_perplexity(f"{example['prompt']}\n{example['completion']}")
            perplexities = np.append(perplexities, perplexity.item())
        
        with open(f"GPT2_Perplexity_{dataset.replace("/","")}.json", "w") as f:
            json.dump(f"Mean GPT2 perplexity for dataset {dataset}: {np.mean(perplexities)}", f)
        print(f"Mean GPT2 perplexity for dataset {dataset}: {np.mean(perplexities)}")
        print(f"Std_Dev GPT2 perplexity for dataset {dataset}: {np.std(perplexities)}")
