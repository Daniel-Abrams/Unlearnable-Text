from unlearnabletext.fine_tuning import finetuneWithLora

def main() -> None:
    finetuneWithLora()

#Fix for path issues: export PYTHONPATH="$PWD/src:$PYTHONPATH"
if __name__ == "__main__":
    main()