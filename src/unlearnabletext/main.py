from fine_tuning import finetuneWithLora
from error_min_generation import fineTuneWithErrorMin
import sys

def main() -> None:
    if sys.argv[1] == "error-min":
        fineTuneWithErrorMin()
    elif sys.argv[1] == "normal":
        finetuneWithLora()

#Fix for path issues: export PYTHONPATH="$PWD/src:$PYTHONPATH"
if __name__ == "__main__":
    main()