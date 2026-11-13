from unlearnabletext.fine_tuning import finetuneWithLora
from unlearnabletext.error_min_generation import fineTuneWithErrorMin

def main() -> None:
    fineTuneWithErrorMin()

#Fix for path issues: export PYTHONPATH="$PWD/src:$PYTHONPATH"
if __name__ == "__main__":
    main()