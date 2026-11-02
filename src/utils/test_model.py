from huggingface_hub import login
import transformers
import torch
import os

hf_token = os.getenv("HF_TOKEN")
login(hf_token)

model_id = "meta-llama/Meta-Llama-3-8B"

pipeline = transformers.pipeline(
    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)
print("Model loaded successfully!")
print(pipeline.model.hf_device_map)
pipeline("Hey how are you doing today?")