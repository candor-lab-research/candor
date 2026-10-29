"""Refusal directions from residuals."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def difference_of_means(harmful: Tensor, harmless: Tensor, orthogonalize: bool = True) -> Tensor:
    """One unit direction per layer, pointing from the harmless mean to the harmful mean.

    With orthogonalize, the part along the mean harmless residual is removed first, so the edit
    leaves alone what ordinary prompts rely on.
    """
    harmful_mean = harmful.double().mean(0)
    harmless_mean = harmless.double().mean(0)
    directions = F.normalize(harmful_mean - harmless_mean, dim=-1)
    if orthogonalize:
        good = F.normalize(harmless_mean, dim=-1)
        directions = directions - (directions * good).sum(-1, keepdim=True) * good
    return F.normalize(directions, dim=-1).float()


def after_layer(directions: Tensor, layer: float) -> Tensor:
    """The direction read after a fractional layer, interpolated between its neighbours."""
    low = math.floor(layer)
    if layer == low or low + 1 >= len(directions):
        return directions[low]
    return F.normalize(torch.lerp(directions[low], directions[low + 1], layer - low), dim=0)
