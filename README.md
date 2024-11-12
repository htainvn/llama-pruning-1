# Llama-Pruning

This project provides tools to load and prune large language models using a structured pruning method. The pruning method is based on the work of Pere Martra with modifications by Mariusz Kurman.

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
python src/main.py --model_name <model_name> --prune_percent <prune_percent> --dtype <dtype> --cache_dir <cache_dir> --device <device> --output <output> [--apply_chat_template]
```

## Arguments

- `--model_name`: Name of the model to load (default: `meditsolutions/Llama-3.2-SUN-2.5B-chat`).
- `--prune_percent`: Percentage of neurons to prune (default: `0.2`).
- `--dtype`: Data type to use (default: `float32`).
- `--cache_dir`: Directory to cache the model (default: `None`).
- `--device`: Device to use (default: `cuda`).
- `--output`: Directory to save the pruned model (default: `pruned_model`).
- `--apply_chat_template`: Apply chat template to the model. (default: `None`)

## Example

```sh
python src/main.py --model_name meditsolutions/Llama-3.2-SUN-2.5B-chat --prune_percent 0.2 --dtype torch.float32 --cache_dir ./cache --device cuda --output ./pruned_model --apply_chat_template
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.