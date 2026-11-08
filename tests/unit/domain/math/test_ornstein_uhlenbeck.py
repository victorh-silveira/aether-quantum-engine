"""Testes unitarios para estimativa e distancia elastica de Ornstein-Uhlenbeck."""

from src.domain.math.ornstein_uhlenbeck import (
    compute_elastic_distance_ou,
    estimate_ornstein_uhlenbeck_parameters,
)


def test_estimate_ou_insufficient_samples():
    theta, mu, sigma = estimate_ornstein_uhlenbeck_parameters([])
    assert theta == 0.0 and mu == 0.0 and sigma == 0.0
    theta, mu, sigma = estimate_ornstein_uhlenbeck_parameters([100.0, 101.0, 102.0])
    assert theta == 0.0 and mu == 102.0 and sigma == 0.0


def test_estimate_ou_constant_series():
    theta, mu, sigma = estimate_ornstein_uhlenbeck_parameters([100.0] * 10)
    assert theta == 0.0
    assert mu == 100.0
    assert sigma == 0.0


def test_estimate_ou_mean_reverting():
    prices = [100.0, 102.0, 99.0, 101.0, 100.0, 98.0, 101.0, 99.5, 100.5, 100.0]
    theta, mu, sigma = estimate_ornstein_uhlenbeck_parameters(prices)
    assert theta > 0.0
    assert 98.0 <= mu <= 102.0
    assert sigma > 0.0


def test_estimate_ou_trending_fallback():
    prices = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    theta, mu, sigma = estimate_ornstein_uhlenbeck_parameters(prices)
    assert theta == 0.0
    assert mu > 0.0
    assert sigma > 0.0


def test_compute_elastic_distance_ou_extremes():
    assert compute_elastic_distance_ou([100.0, 101.0]) == 0.0
    assert compute_elastic_distance_ou([100.0] * 10) == 0.0

    prices_high = [100.0, 101.0, 99.0, 100.0, 99.5, 100.5, 99.0, 100.0, 101.0, 108.0]
    zeta_high = compute_elastic_distance_ou(prices_high)
    assert zeta_high > 2.0

    prices_low = [100.0, 101.0, 99.0, 100.0, 99.5, 100.5, 99.0, 100.0, 101.0, 92.0]
    zeta_low = compute_elastic_distance_ou(prices_low)
    assert zeta_low < -2.0


def test_compute_elastic_distance_ou_trending():
    prices_trend = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 100.0]
    zeta = compute_elastic_distance_ou(prices_trend)
    assert zeta > 1.0
