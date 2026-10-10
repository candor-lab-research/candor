import pytest
import torch
import torch.nn.functional as F

from candor import kl


@pytest.fixture
def distributions():
    torch.manual_seed(4)
    base = F.log_softmax(torch.randn(8, 50) * 3, dim=-1)
    edited = F.log_softmax(base + 0.3 * torch.randn(8, 50), dim=-1)
    return base, edited


def test_identical_distributions(distributions):
    base, _ = distributions
    assert kl.exact(base, base) == pytest.approx(0.0, abs=1e-7)
    assert kl.mean_from_top(kl.top(base, 10), kl.top(base, 10)) == pytest.approx(0.0, abs=1e-7)


def test_full_lists_give_the_exact_value(distributions):
    base, edited = distributions
    approx = kl.mean_from_top(kl.top(base, 50), kl.top(edited, 50))
    assert approx == pytest.approx(kl.exact(base, edited), rel=1e-4)


def test_converges_as_more_tokens_are_listed(distributions):
    base, edited = distributions
    exact = kl.exact(base, edited)
    errors = [abs(kl.mean_from_top(kl.top(base, k), kl.top(edited, k)) - exact) for k in (5, 20, 45)]
    assert errors[2] < errors[0]
    assert errors[2] < 0.05 * exact
