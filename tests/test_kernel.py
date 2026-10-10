import pytest

from candor.kernel import Kernel

KERNEL = Kernel(max_weight=1.2, max_weight_position=10.0, min_weight=0.4, min_weight_distance=4.0)


def test_peak():
    assert KERNEL.at(10) == pytest.approx(1.2)


def test_falls_linearly_to_min_weight_at_the_edge():
    assert KERNEL.at(12) == pytest.approx(0.8)
    assert KERNEL.at(14) == pytest.approx(0.4)
    assert KERNEL.at(6) == pytest.approx(0.4)


def test_zero_outside_the_band():
    assert KERNEL.at(15) == 0.0
    assert KERNEL.at(5) == 0.0


def test_fractional_peak():
    kernel = Kernel(1.0, 2.5, 0.0, 1.0)
    assert kernel.at(2) == pytest.approx(0.5)
    assert kernel.at(3) == pytest.approx(0.5)
    assert kernel.at(4) == 0.0


def test_zero_distance_edits_only_the_peak():
    kernel = Kernel(1.0, 3.0, 0.5, 0.0)
    assert kernel.at(3) == 1.0
    assert kernel.at(2) == 0.0
