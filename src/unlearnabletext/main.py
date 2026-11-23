from fine_tuning import finetuneWithLora
from evaluate import evaluate_model
from error_min_generation import fineTuneWithErrorMin
from gpt2_perplexity import calculate_GPT2_perplexity
from sbert import sbert_similarity
import sys

def main() -> None:
    mode = sys.argv[1]

    if mode == "error-min":
        fineTuneWithErrorMin()
    elif mode == "perplexity":
        dataset = sys.argv[2]
        calculate_GPT2_perplexity(dataset)
    elif mode == "sbert":
        dataset = sys.argv[2]
        sbert_similarity(dataset)
    else:
        model_id = sys.argv[2]
        if mode == "finetune":
            dataset = sys.argv[3]
            finetuneWithLora(model_id, dataset)
        elif mode == "evaluate":
            evaluate_model(model_id=model_id)
    
if __name__ == "__main__": 
    main()