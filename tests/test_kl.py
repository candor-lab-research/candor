import math

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


def test_unlisted_tokens_worked_out_by_hand():
    base = torch.tensor([[0.5, 0.3, 0.15, 0.05]], dtype=torch.float64).log()
    edited = torch.tensor([[0.4, 0.15, 0.35, 0.1]], dtype=torch.float64).log()
    # With two tokens listed, base shows 0 and 1, edited shows 0 and 2. Token 1 gets edited's
    # smallest listed probability, 0.35. Everything else is one bucket: 1 - 0.5 - 0.3 = 0.2 for base,
    # 1 - 0.4 - 0.35 = 0.25 for edited.
    expected = 0.5 * math.log(0.5 / 0.4) + 0.3 * math.log(0.3 / 0.35) + 0.2 * math.log(0.2 / 0.25)
    assert kl.mean_from_top(kl.top(base, 2), kl.top(edited, 2)) == pytest.approx(expected, rel=1e-9)


def test_twenty_tokens_land_near_the_exact_value_on_a_peaked_distribution():
    torch.manual_seed(6)
    vocab = 32_000
    ranks = torch.arange(1, vocab + 1, dtype=torch.float64)
    base = F.log_softmax(-3 * ranks.log() + 0.3 * torch.randn(16, vocab, dtype=torch.float64), dim=-1)
    edited = F.log_softmax(base + 0.3 * torch.randn(16, vocab, dtype=torch.float64), dim=-1)
    # Like a chat model's first token: the top 20 hold about 99.9% of the probability.
    assert base.exp().topk(20).values.sum(-1).mean() == pytest.approx(0.999, abs=5e-4)
    approx = kl.mean_from_top(kl.top(base, 20), kl.top(edited, 20))
    assert approx == pytest.approx(kl.exact(base, edited), rel=0.01)
