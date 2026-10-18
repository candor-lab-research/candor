"""The weight edit, as LoRA factors."""

from enum import StrEnum

import torch
import torch.nn.functional as F
from torch import Tensor


class RowNorm(StrEnum):
    NONE = "none"
    PRE = "pre"
    FULL = "full"


# The rank heretic approximates a "full" edit at (its default full_normalization_lora_rank).
FULL_RANK = 3


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
    if row_norm is RowNorm.FULL:
        raise ValueError('a "full" edit isn\'t linear in strength: build it at its strength with full()')
    return -(w.norm(dim=1) * r)[:, None], (r @ F.normalize(w, dim=1))[None, :]


def full(
    weight: Tensor, direction: Tensor, strength: float, rank: int = FULL_RANK, seed: int = 0
) -> tuple[Tensor, Tensor]:
    """Factors b (out, rank) and a (rank, in) of a FULL edit at this strength, so that weight + b @ a is
    the edited weight, as heretic builds it: the direction removed from the row-normalised weights, each
    row normalised again and scaled back to its original length, and the change approximated at rank
    by a randomised SVD seeded with seed, which keeps the row lengths only roughly. The renormalising
    makes it not linear in strength, so each strength needs factors of its own."""
    w = weight.float()
    r = direction.float().to(w.device)
    lengths = w.norm(dim=1, keepdim=True)
    unit = F.normalize(w, dim=1)
    edited = F.normalize(unit - strength * r[:, None] * (r @ unit)[None, :], dim=1) * lengths
    torch.manual_seed(seed)
    u, s, v = torch.svd_lowrank(edited - w, q=2 * rank + 4, niter=6)
    root = s[:rank].sqrt()
    return u[:, :rank] * root, (v[:, :rank] * root).T


def edit_factors(
    weight: Tensor, direction: Tensor, strength: float, row_norm: RowNorm
) -> tuple[Tensor, Tensor]:
    """Factors (b, a) of the change an edit at this strength makes, so that weight + b @ a is the
    edited weight, on the CPU. A "full" change takes an SVD, so it is built on the GPU when there is
    one; a rank-one change on the CPU, as Run.adapter_set builds it. Either way the adapters scored
    and the file exported hold one change on any one machine."""
    if row_norm is RowNorm.FULL:
        b, a = full(weight.to("cuda" if torch.cuda.is_available() else "cpu"), direction, strength)
        return b.cpu(), a.cpu()
    b, a = rank_one(weight, direction, row_norm)
    return strength * b, a
