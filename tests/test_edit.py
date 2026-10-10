import torch
import torch.nn.functional as F

from candor.edit import RowNorm, rank_one


def random_case(out_features=12, in_features=20):
    torch.manual_seed(1)
    return torch.randn(out_features, in_features), F.normalize(torch.randn(out_features), dim=0)


def test_shapes():
    weight, direction = random_case()
    for row_norm in RowNorm:
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


def test_pre_scales_each_row_by_its_length():
    weight, direction = random_case()
    b, a = rank_one(weight, direction, RowNorm.PRE)
    normalised = F.normalize(weight, dim=1)
    expected = -weight.norm(dim=1, keepdim=True) * torch.outer(direction, direction @ normalised)
    torch.testing.assert_close(b @ a, expected)


def test_bfloat16_weights_give_float32_factors():
    weight, direction = random_case()
    b, a = rank_one(weight.bfloat16(), direction)
    assert b.dtype == a.dtype == torch.float32
