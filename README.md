# Llama-Pruning

This project provides tools to load and prune large language models using a structured pruning method. The pruning method is based on the work of [Pere Martra](https://github.com/peremartra) with modifications by [Mariusz Kurman](https://github.com/mkurman).

This method is applicable to all models with a Llama-like architecture that includes MLP gating, such as Llama, Phi, Mistral, Qwen, SmolLM, and others.

Blog post: [Model Pruning: A New Approach](https://mkurman.substack.com/p/model-pruning-a-new-approach).

Original work: [Large-Language-Model-Notebooks-Course](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/6-PRUNING/6_3_pruning_structured_llama3.2-1b_OK.ipynb).

Pere Martra's book: [Large Language Models: Apply and Implement Strategies for Large Language Models](https://amzn.to/4eanT1g)

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Arguments](#arguments)
- [Example](#example)
- [License](#license)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/MedITSolutionsKurman/llama-pruning.git
    cd llama-pruning
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

To load and prune a model, run the `main.py` script with the appropriate arguments:

```sh
python src/main.py --model_name <model_name> --prune_percent <prune_percent> --dtype <dtype> --cache_dir <cache_dir> --device <device> --output <output> --prompt <prompt> --max_new_tokens <max_new_tokens> [--apply_chat_template]
```

## Arguments

- `--model_name`: Name of the model to load (default: `meditsolutions/Llama-3.2-SUN-2.5B-chat`).
- `--prune_percent`: Percentage of MLP neurons to prune (default: `0.2`).
- `--dtype`: Data type to use (default: `float32`).
- `--cache_dir`: Directory to cache the model (default: `None`).
- `--device`: Device to use (default: `cuda`).
- `--output`: Directory to save the pruned model (default: `pruned_model`).
- `--apply_chat_template`: Apply chat template to the model (default: `None`).
- `--prompt`: Prompt to generate the output (default: `What is the capital of France?`).
- `--max_new_tokens`: Maximum number of tokens to generate (default: `50`).
- `--target_size`: Target size for the MLPs intermediate layer. (`prune_percent` will be ignored).
- `--prune_method`: Method to use for pruning. Currently, only "mk_prune" (alias: "mk") and "mk_prune_adjusted" (alias: "mka") are supported. (default: `mk_prune`)
- `--use_normalized_weights`: Use normalized weights to calculate the final weights. (default: `False`)
- `--use_layer_norm_tweaks`: Apply layer normalization changes to account for the impact of pruned neurons. (default: `False`)
- `--layer_norm_scale`: Layer normalization scale. Only used if use_layer_norm_tweaks is True. (default: 4.0)
- `--test_only`: Run the test only. Do not save the model (default: `False`).
- `--print_summary`: Print the pruned model summary. (default: `False`)

## Example

```sh
python src/main.py --model_name meditsolutions/Llama-3.2-SUN-2.5B-chat --prune_percent 0.2 --dtype torch.float32 --cache_dir ./cache --device cuda --output ./pruned_model --prompt "How to prepare pierogi (famous Polish dish)?" --max_new_tokens 128 --apply_chat_template
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
