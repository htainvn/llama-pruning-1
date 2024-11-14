import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import warnings

from src.model.load import load_model
from src.func.update import update_model
from src.utils.count import count_parameters
from src.model.generate import get_output
from argparse import ArgumentParser

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # Parse the arguments.
    parser = ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="meditsolutions/Llama-3.2-SUN-2.5B-chat",
        help="Name of the model to load.",
    )
    parser.add_argument(
        "--prune_percent",
        type=float,
        default=0.2,
        help="Percentage of MLP neurons to prune.",
    )
    parser.add_argument(
        "--dtype", type=str, default="torch.float32", help="Data type to use."
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None, help="Directory to cache the model."
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use.")
    parser.add_argument(
        "--output",
        type=str,
        default="pruned_model",
        help="Directory to save the pruned model.",
    )
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        default=False,
        help="Apply chat template to the model.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What is the capital of France?",
        help="Prompt to generate the output.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=None,
        help="Target size for the MLPs intermediate layer. (prune_percent will be ignored)",
    )
    parser.add_argument(
        "--test_only",
        action="store_true",
        default=False,
        help="Run the test only. Do not save the model.",
    )
    parser.add_argument(
        "--prune_method",
        type=str,
        default="mk_prune",
        help='Method to use for pruning. Currently, only "mk_prune" (alias: "mk") and "mk_prune_adjusted" (alias: "mka") are supported. (default: mk_prune)',
    )

    parser.add_argument(
        "--use_normalized_weights",
        action="store_true",
        default=False,
        help="Use normalized weights to calculate the final weights.",
    )

    parser.add_argument(
        "--use_layer_norm_tweaks",
        action="store_true",
        default=False,
        help="Apply layer normalization changes to account for the impact of pruned neurons.",
    )

    parser.add_argument(
        "--layer_norm_scale",
        type=float,
        default=4.0,
        help="Layer normalization scale. Only used if use_layer_norm_tweaks is True. (default: 4.0)",
    )

    parser.add_argument(
        "--print_summary",
        action="store_true",
        default=False,
        help="Print the pruned model summary.",
    )

    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"

    print(f"\nUsing device: {device}")

    if args.prune_method not in ["mk_prune", "mk", "mk_prune_adjusted", "mka"]:
        raise ValueError(f"Unknown prune method: {args.prune_method}")
    elif args.prune_method in ["mk", "mk_prune"]:
        args.prune_method = "mk_prune"
    elif args.prune_method in ["mka", "mk_prune_adjusted"]:
        args.prune_method = "mk_prune_adjusted"
    else:
        pass

    # Load the model and tokenizer.
    model, tokenizer = load_model(
        model_name=args.model_name,
        dtype=dtype,
        cache_dir=args.cache_dir,
        device=args.device,
    )

    # Get sample output from the model.

    print("\nGenerating output from the original model...")
    output, logits_original = get_output(
        args.prompt,
        model,
        tokenizer,
        device=device,
        apply_chat_template=args.apply_chat_template,
        max_new_tokens=args.max_new_tokens,
    )

    parameters_before = count_parameters(model)

    # Prune the model.

    if args.target_size is not None:
        print(
            f"\nPruning the model to target size: {args.target_size}, ignoring prune_percent."
        )

    model = update_model(
        model,
        args.prune_percent,
        prune_method=args.prune_method,
        use_normalized_weights=args.use_normalized_weights,
        use_layer_norm_tweaks=args.use_layer_norm_tweaks,
        layer_norm_scale=args.layer_norm_scale,
        device=args.device,
        target_size=args.target_size,
    )

    parameters_after = count_parameters(model)
    reduction = (parameters_before - parameters_after) / parameters_before * 100

    print(
        f"\nModel pruned successfully. Parameters before: {parameters_before}, Parameters after: {parameters_after}, Reduction: {reduction:.2f}%."
    )
    print("\nGenerating output from the pruned model...")

    output_pruned, logits_pruned = get_output(
        args.prompt,
        model,
        tokenizer,
        device=device,
        apply_chat_template=args.apply_chat_template,
        max_new_tokens=args.max_new_tokens,
    )

    # Calculate the difference between the logits of the original and pruned model using KL divergence.
    diff = torch.nn.functional.kl_div(
        logits_original.log_softmax(dim=-1),
        logits_pruned.softmax(dim=-1),
        reduction="batchmean",
    ).item() / logits_original.size(1)
    print(
        f"\nKL divergence between the logits of the original and pruned model: {diff:.2f}. \n\nAbout: KL divergence is a measure of how one probability distribution differs from a baseline distribution. In this case, it is used to measure the difference between the logits of the original and pruned model. The lower the value, the more similar the distributions are."
    )

    original_max = logits_original.max().item()
    pruned_max = logits_pruned.max().item()

    original_min = logits_original.min().item()
    pruned_min = logits_pruned.min().item()

    original_mean = logits_original.mean().item()
    pruned_mean = logits_pruned.mean().item()

    print(
        f"\nLogits statistics:\nOriginal: Max: {original_max:.2f}, Min: {original_min:.2f}, Mean: {original_mean:.2f}\nPruned: Max: {pruned_max:.2f}, Min: {pruned_min:.2f}, Mean: {pruned_mean:.2f}"
    )

    if args.print_summary:
        print(f"\nModel summary after pruning:\n{model}")

    if args.test_only:
        print("\nTest completed successfully.")
        exit()

    print(f"\nSaving the pruned model at {args.output}...")
    model = model.to(dtype)
    model.save_pretrained(args.output)

    print(f"\nModel saved at {args.output}.")
