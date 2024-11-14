import torch
import torch.nn as nn
from typing import Optional
from tqdm import tqdm

from copy import deepcopy

from src.method.mk_prune import prune_neuron_pairs


# Iterates throught the model layers and applies pruning.
# Note: This method was previously copied from the source given below:
# https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/6-PRUNING/6_3_pruning_structured_llama3.2-1b_OK.ipynb
# It was modified to include new ways to calculate the importance score, include the target_size parameter and take normalizations into account.
def update_model(
    model: nn.Module,
    prune_percent: float,
    method: str = "mk_prune",
    device: Optional[str] = "cuda",
    target_size: Optional[int] = None,
) -> nn.Module:
    """
    It modifies each mlp layer present in model, to retain only the most
    important neurons. Creating new smaller versions of each layer pruned.

    Args:
    - model: Model to prune.
    - prune_percent: Percentage of neurons to prune.
    - device: Device to use.
    - target_size: Target size for the intermediate layer. (prune_percent will be ignored)

    Returns:
    - model: New pruned model.
    """
    new_intermediate_size = None

    # final_model = deepcopy(model)

    # loop for each model layer.
    for idx, layer in tqdm(
        enumerate(model.model.layers), total=len(model.model.layers)
    ):
        # Since each layer is a LlamaDecoderLayer it contains multiple components
        # Attention, MLP and Layer norms. We're targetting MLP component
        # by accesing layer.mlp.
        mlp = layer.mlp

        if method != "mk_prune":
            raise ValueError(f"Unknown method: {method}")

        # Call the prune_neiron_pairs with the layers and receiving the pruned.
        new_gate_proj, new_up_proj, new_down_proj, new_size = prune_neuron_pairs(
            mlp, prune_percent, device=device, target_size=target_size
        )

        if idx < len(model.model.layers) - 1:
            last_layer_original_sum = (
                torch.abs(mlp.down_proj.weight.data).sum()
                + torch.abs(mlp.up_proj.weight.data).sum()
                + torch.abs(mlp.gate_proj.weight.data).sum()
            )
            last_layer_pruned_sum = (
                torch.abs(new_down_proj.weight.data).sum()
                + torch.abs(new_up_proj.weight.data).sum()
                + torch.abs(new_gate_proj.weight.data).sum()
            )

            # Update the next layer normalization weights.
            model.model.layers[idx + 1].input_layernorm.weight.data *= (
                1.0
                + (1.0 - torch.abs(last_layer_pruned_sum / last_layer_original_sum)) / 9
            )

        # Replace the Origiginal Layers with Pruned Layers.
        mlp.gate_proj = new_gate_proj
        mlp.up_proj = new_up_proj
        mlp.down_proj = new_down_proj

        torch.cuda.empty_cache()

        # new_intermediate_size only needs to be set once
        if new_intermediate_size is None:
            new_intermediate_size = new_size

    # Update the last layer normalization weights.
    model.model.norm.weight.data *= (
        1.0 + (1.0 - torch.abs(last_layer_pruned_sum / last_layer_original_sum)) / 9
    )

    # Update the model config file.
    model.config.intermediate_size = new_intermediate_size

    # return final_model
    return model
