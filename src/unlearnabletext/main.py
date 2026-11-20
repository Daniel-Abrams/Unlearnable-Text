from fine_tuning import finetuneWithLora
from evaluate import evaluate_model
from error_min_generation import fineTuneWithErrorMin
import sys

def main() -> None:
    mode = sys.argv[1]

    if mode == "error-min":
        fineTuneWithErrorMin()
    else:
        model_id = sys.argv[2]
        local_models = ["meta-llamaMeta-Llama-3-8B_allenaiai2_arc_lora_adapter", 
                        "meta-llamaMeta-Llama-3-8B_daniel-abramsarc_challenge_perturbed_lora_adapter"
                    ]
        if model_id in local_models:
            local = True
        else:
            local = False
        if mode == "finetune":
            dataset = sys.argv[3]
            finetuneWithLora(model_id, dataset)
        elif mode == "evaluate":
            evaluate_model(model_id=model_id, local=local)
    
#Fix for path issues: export PYTHONPATH="$PWD/src:$PYTHONPATH"
if __name__ == "__main__": 
    main()