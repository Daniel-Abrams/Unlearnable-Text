from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import numpy as np


def sbert_similarity(dataset2):
    
    similarity_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    
    d1 = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    d2 = load_dataset(dataset2, split="train")
    
    similarities = np.array([])
    
    for q, p in zip(d1,d2):
        emb1 = similarity_model.encode(q['question'])
        emb2 = similarity_model.encode(p['question'])
        score = similarity_model.similarity(emb1, emb2)
        similarities = np.append(similarities, score)
    
    print(f"Mean SBERT similarity: {np.mean(similarities)}")
    print(f"Std_Dev SBERT similarity {np.std(similarities)}")



dataset2 = "daniel-abrams/arc_challenge_answer_tagged"
sbert_similarity(dataset2)
