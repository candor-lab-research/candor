import pytest
import torch
import torch.nn.functional as F

from candor.edit import RowNorm, edit_factors, full, rank_one


def random_case(out_features=12, in_features=20):
    torch.manual_seed(1)
    return torch.randn(out_features, in_features), F.normalize(torch.randn(out_features), dim=0)


def test_shapes():
    weight, direction = random_case()
    for row_norm in (RowNorm.NONE, RowNorm.PRE):
        b, a = rank_one(weight, direction, row_norm)
        assert b.shape == (12, 1)
        assert a.shape == (1, 20)


def test_none_at_strength_one_removes_the_direction_from_the_output():
    weight, direction = random_case()
    b, a = rank_one(weight, direction, RowNorm.NONE)
    edited = weight + b @ a
    x = torch.randn(20, 5)
    torch.testing.assert_close(direction @ edited @ x, torch.zeros(5), atol=1e-5, rtol=0)


def test_pre_equals_none_when_every_row_has_the_same_length():
    weight, direction = random_case()
    weight = 3.0 * F.normalize(weight, dim=1)
    b_none, a_none = rank_one(weight, direction, RowNorm.NONE)
    b_pre, a_pre = rank_one(weight, direction, RowNorm.PRE)
    torch.testing.assert_close(b_pre @ a_pre, b_none @ a_none)


def test_pre_takes_strength_times_the_direction_out_of_the_normalised_rows():
    # With every row divided by its original length, strength s leaves (1 - s) of what the weight
    # writes along the direction and changes nothing orthogonal to it.
    weight, direction = random_case()
    b, a = rank_one(weight, direction, RowNorm.PRE)
    lengths = weight.norm(dim=1, keepdim=True)
    before = weight / lengths
    orthogonal = torch.eye(len(direction)) - torch.outer(direction, direction)
    for strength in (0.5, 1.0, 1.5):
        after = (weight + strength * b @ a) / lengths
        torch.testing.assert_close(direction @ after, (1 - strength) * (direction @ before))
        torch.testing.assert_close(orthogonal @ after, orthogonal @ before)


def test_bfloat16_weights_give_float32_factors():
    weight, direction = random_case()
    b, a = rank_one(weight.bfloat16(), direction)
    assert b.dtype == a.dtype == torch.float32


def heretics_full(weight, direction, strength, rank, seed):
    """heretic's own steps for row_normalization "full" (heretic/model.py, 1.4.0), as it builds a
    trial's LoRA factors."""
    w_org = weight.float()
    norms = torch.linalg.vector_norm(w_org, dim=1, keepdim=True)
    w = F.normalize(w_org, p=2, dim=1)
    lora_a = (direction @ w).view(1, -1)
    lora_b = (-strength * direction).view(-1, 1)
    w = F.normalize(w + lora_b @ lora_a, p=2, dim=1) * norms - w_org
    torch.manual_seed(seed)
    u, s, v = torch.svd_lowrank(w, q=2 * rank + 4, niter=6)
    root = torch.sqrt(s[:rank])
    return u[:, :rank] @ torch.diag(root), torch.diag(root) @ v[:, :rank].T


def test_full_builds_heretics_factors():
    torch.manual_seed(1)
    weight = torch.randn(64, 48)
    direction = F.normalize(torch.randn(64), dim=0)
    b, a = full(weight, direction, 1.37, 3, 2521449329)
    hb, ha = heretics_full(weight, direction, 1.37, 3, 2521449329)
    assert b.shape == (64, 3) and a.shape == (3, 48)
    torch.testing.assert_close(b, hb)
    torch.testing.assert_close(a, ha)


def test_full_keeps_each_rows_length_and_isnt_linear_in_strength():
    torch.manual_seed(2)
    # A weight of rank 1 makes a change of rank at most 2, which the factors then hold whole.
    weight = torch.outer(torch.randn(40), torch.randn(12))
    direction = F.normalize(torch.randn(40), dim=0)
    b, a = full(weight, direction, 1.2)
    torch.testing.assert_close((weight + b @ a).norm(dim=1), weight.norm(dim=1))
    weight = torch.randn(40, 12)
    once, twice = (edit_factors(weight, direction, s, RowNorm.FULL) for s in (1.0, 2.0))
    doubled = 2 * (once[0] @ once[1])
    assert (twice[0] @ twice[1] - doubled).norm() > 0.05 * doubled.norm()


def test_edit_factors_fold_a_linear_edits_strength_in():
    torch.manual_seed(3)
    weight = torch.randn(16, 8)
    direction = F.normalize(torch.randn(16), dim=0)
    for row_norm in (RowNorm.NONE, RowNorm.PRE):
        b, a = rank_one(weight, direction, row_norm)
        fb, fa = edit_factors(weight, direction, 0.7, row_norm)
        torch.testing.assert_close(fb @ fa, 0.7 * (b @ a))


def test_a_full_edit_has_no_single_adapter_for_every_strength():
    with pytest.raises(ValueError, match="isn't linear in strength"):
        rank_one(torch.ones(4, 4), torch.ones(4), RowNorm.FULL)
