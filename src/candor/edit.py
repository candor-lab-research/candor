"""The weight edit, as rank-1 LoRA factors."""

from enum import StrEnum

import torch.nn.functional as F
from torch import Tensor


class RowNorm(StrEnum):
    NONE = "none"
    PRE = "pre"


def rank_one(weight: Tensor, direction: Tensor, row_norm: RowNorm = RowNorm.PRE) -> tuple[Tensor, Tensor]:
    """Factors b (out, 1) and a (1, in), so that weight + strength * b @ a is the edited weight.

    NONE removes the direction from what the layer writes: delta = -r (rᵀW).
    PRE computes the same edit on row-normalised weights, then scales each row back to its
    original length. Both are linear in strength, so one adapter covers every strength.
    """
    w = weight.float()
    r = direction.float().to(w.device)
    if row_norm is RowNorm.NONE:
        return -r[:, None], (r @ w)[None, :]
    return -(w.norm(dim=1) * r)[:, None], (r @ F.normalize(w, dim=1))[None, :]
