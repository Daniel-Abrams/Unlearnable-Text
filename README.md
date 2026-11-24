# Unlearnable Text

## Overview
In this work, we attempt to generate unlearnable noise for text. We use the ARC-Challenge benchmark for creating and 
testing the effectiveness of unlearnable noise. Read the [paper](Abrams_Ssennya_Unlearnable_Perturbations.pdf) for more details.

## Requirements
- Python 3.x
- Mamba
- Dependencies listed in `environment.yml`

## Setup

Clone the repository:

```bash
git clone https://github.com/Daniel-Abrams/Unlearnable-Text
cd Unlearnable-Text
```

Load the Mamba module:

```bash
ml mamba
```

Create the environment:

```bash
mamba env create -n unlearnable-text --file environment.yml
```

Activate the environment:

```bash
source activate unlearnable-text
```

## Usage

### Generating and Finetuning on Datasets
Most scripts can be run with main.py.

To generate the unlearnable noise with `Meta-Llama-3-8B`, run

Run the project using Python:

```bash
python src/unlearnabletext/main.py error-min
```

To finetune a model on a dataset, run
```bash
python src/unlearnabletext/main.py finetune <name-of-model> <name-of-dataset>
```
The model names you should use are either `meta-llama/Meta-Llama-3-8B` or `Qwen/Qwen3-8B`.

The valid dataset names you can use are `allenai/ai2_arc`, `daniel-abrams/arc_challenge_perturbed`, `daniel-abrams/arc_challenge_random_edits`, `daniel-abrams/arc_challenge_answer_tagged`.

If you'd like to generate the tagged or randomly edited datasets yourself, run `one_token_tag_dataset.py` and `one_token_tag_dataset.py`

### Evaluating a Model

Fintuning will create a model checkpoint with the name of the model (+ LoRA adpater) and the dataset it was trained on, stripped of any forward slashes. For example, `meta-llamaMeta-Llama-3-8B_daniel-abramsarc_challenge_perturbed_lora_adapter`. You can run 

To evaluate a model on the ARC-Challenge test split, run
```bash
python src/unlearnabletext/main.py evaluate <name-of-model>
```
and paste the name of said model as an argument. For zero-shot evaluation, pass `meta-llama/Meta-Llama-3-8B` or `Qwen/Qwen3-8B`.

### Calculating GPT2 Perplexity 
To calculate the perplexity of a dataset, run

```bash
python src/unlearnabletext/main.py perplexity <name-of-dataset>
```

### Calculating SBERT Similarity
To calculate the SBERT similarity of a dataset in comparison to the original ARC-Challenge train split, run

```bash
python src/unlearnabletext/main.py sbert <name-of-dataset>
```

## Project Structure

```text
Unlearnable-Text/
├── src/                 # Source code
├── environment.yml      # Environment dependencies
└── README.md
```

## Contributors

Daniel Abrams   <dlabram4@asu.edu> \
Marc Sennya     <mssennya@asu.edu>
