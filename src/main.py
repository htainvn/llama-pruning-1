import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch

from src.model.load import load_model
from method.mk_prune import update_model
from src.utils.count import count_parameters
from src.model.generate import get_output
from argparse import ArgumentParser

if __name__ == '__main__':
    # Parse the arguments.
    parser = ArgumentParser()

    parser.add_argument('--model_name', type=str, default='meditsolutions/Llama-3.2-SUN-2.5B-chat', help='Name of the model to load.')
    parser.add_argument('--prune_percent', type=float, default=0.2, help='Percentage of neurons to prune.')
    parser.add_argument('--dtype', type=str, default='torch.float32', help='Data type to use.')
    parser.add_argument('--cache_dir', type=str, default=None, help='Directory to cache the model.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use.')
    parser.add_argument('--output', type=str, default='pruned_model', help='Directory to save the pruned model.')
    parser.add_argument('--apply_chat_template', action="store_true", default=False, help='Apply chat template to the model.')

    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    device = 'cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")

    # Load the model and tokenizer.
    model, tokenizer = load_model(model_name=args.model_name, dtype=dtype, cache_dir=args.cache_dir, device=args.device)

    # Get sample output from the model.
    prompt = "What is the capital of France?"

    print("Generating output from the original model...")
    output = get_output(prompt, model, tokenizer, device=device, apply_chat_template=args.apply_chat_template)

    parameters_before = count_parameters(model)

    # Prune the model.
    model = update_model(model, args.prune_percent, args.device)

    parameters_after = count_parameters(model)
    reduction = (parameters_before - parameters_after) / parameters_before * 100

    print(f'Model pruned successfully. Parameters before: {parameters_before}, Parameters after: {parameters_after}, Reduction: {reduction:.2f}%.')
    print("Generating output from the pruned model...")

    output_pruned = get_output(prompt, model, tokenizer, device=device, apply_chat_template=args.apply_chat_template)

    model = model.to(dtype).cpu()
    model.save_pretrained(args.output)

    print(f'Model saved at {args.output}.')