import torch


# Adjusted Importance:
# The adjusted importance score is calculated by dividing the sum of the maximum absolute weight,
# the mean of the weights, and the maximum absolute weight of the weights that are less than the mean,
# by the number of weights that are less than the mean.
def get_importance(weight: torch.Tensor) -> torch.Tensor:
    """
    Calculate the adjusted importance score for the weight matrix.

    Args:
    - weight: Weight matrix from the layer.

    Returns:
    - adjusted_importance: Adjusted importance score for the weight matrix.
    """
    mask = weight > 0.0
    positive_max = (weight * mask.float()).max(dim=1).values
    negative_max_abs = (torch.abs(weight * (~mask).float())).max(dim=1).values

    weight_adjusted_mean = (
        torch.abs(weight).sum(dim=1) - positive_max - negative_max_abs
    ) / (weight.size(1) - 2)

    weight_small_count = (torch.abs(weight) <= weight_adjusted_mean.unsqueeze(1)).sum(
        dim=1
    )

    adjusted_importance = (
        (positive_max.float() + negative_max_abs.float() + weight_adjusted_mean.float())
        * weight.size(1)
        / weight_small_count
    )

    return adjusted_importance
