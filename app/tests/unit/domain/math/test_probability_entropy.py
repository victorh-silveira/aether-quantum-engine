"""Testes de entropia binaria e penalizacao."""

from src.domain.math.probability_entropy import binary_entropy, entropy_penalty_factor


def test_binary_entropy_symmetry():
    h_half = binary_entropy(0.5)
    assert h_half > binary_entropy(0.9)
    assert binary_entropy(0.5, base="2") == 1.0


def test_entropy_penalty_factor_bounds():
    assert entropy_penalty_factor(0.999, ceiling=0.92, floor=0.0) < 0.05
    assert entropy_penalty_factor(0.5, ceiling=0.1, floor=0.0) == 1.0
    mid = entropy_penalty_factor(0.7, ceiling=1.0, floor=0.0)
    assert 0.0 < mid < 1.0
