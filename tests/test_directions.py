import torch
import torch.nn.functional as F

from candor.directions import after_layer, difference_of_means


def synthetic(signal: torch.Tensor, prompts=500, layers=3, hidden=16):
    torch.manual_seed(2)
    harmless = torch.randn(prompts, layers, hidden) + 2.0
    harmful = torch.randn(prompts, layers, hidden) + 2.0 + 4.0 * signal
    return harmful, harmless


def test_finds_the_planted_direction():
    signal = F.normalize(torch.randn(16), dim=0)
    harmful, harmless = synthetic(signal)
    directions = difference_of_means(harmful, harmless, orthogonalize=False)
    assert directions.shape == (3, 16)
    assert torch.all(directions @ signal > 0.98)


def test_orthogonalized_directions_ignore_the_harmless_mean():
    harmful, harmless = synthetic(F.normalize(torch.randn(16), dim=0))
    directions = difference_of_means(harmful, harmless)
    good = F.normalize(harmless.mean(0), dim=-1)
    torch.testing.assert_close((directions * good).sum(-1), torch.zeros(3), atol=1e-5, rtol=0)
    torch.testing.assert_close(directions.norm(dim=-1), torch.ones(3))


def test_after_layer():
    directions = F.normalize(torch.randn(5, 8), dim=-1)
    torch.testing.assert_close(after_layer(directions, 1.0), directions[2])
    middle = after_layer(directions, 1.5)
    torch.testing.assert_close(middle, F.normalize(directions[2] + directions[3], dim=0))
    torch.testing.assert_close(after_layer(directions, 3.0), directions[4])


def test_a_whole_layer_gives_that_layers_direction_as_it_is():
    # Not normalised, so any arithmetic on the way would show; a search's adapters for a layer and a
    # measurement's of the same trial must be built from identical directions.
    found = torch.randn(6, 16, generator=torch.Generator().manual_seed(1))
    assert torch.equal(after_layer(found, 3), found[4])
    assert torch.equal(after_layer(found, 3.0), found[4])
