"""Estimativa de reversao a media estocastica via processo de Ornstein-Uhlenbeck."""

from __future__ import annotations

import math
from collections.abc import Sequence


def estimate_ornstein_uhlenbeck_parameters(
    prices: Sequence[float],
) -> tuple[float, float, float]:
    """Estima parametros (theta, mu, sigma) de Ornstein-Uhlenbeck via regressao AR(1)."""
    n = len(prices)
    if n < 5:
        return 0.0, float(prices[-1]) if n > 0 else 0.0, 0.0
    x_vals = [float(prices[i]) for i in range(n - 1)]
    y_vals = [float(prices[i]) for i in range(1, n)]
    m = n - 1
    mean_x = sum(x_vals) / m
    mean_y = sum(y_vals) / m
    var_x = sum((xi - mean_x) ** 2 for xi in x_vals) / m
    if var_x < 1e-14:
        return 0.0, mean_y, 0.0
    cov_xy = sum((x_vals[i] - mean_x) * (y_vals[i] - mean_y) for i in range(m)) / m
    a = cov_xy / var_x
    b = mean_y - a * mean_x
    residuals = [y_vals[i] - (a * x_vals[i] + b) for i in range(m)]
    sse = sum(r**2 for r in residuals)
    sigma_eps = math.sqrt(max(0.0, sse / max(1, m - 2)))
    if a < 0.999999:
        theta = -math.log(max(1e-4, a)) if a > 1e-4 else max(0.1, 1.0 - a)
        mu = b / max(1e-6, 1.0 - a)
        denom = 1.0 - math.exp(-2.0 * min(5.0, theta))
        sigma = sigma_eps * math.sqrt((2.0 * theta) / max(1e-12, denom))
        return theta, mu, sigma
    var_y = sum((yi - mean_y) ** 2 for yi in y_vals) / m
    return 0.0, mean_y, math.sqrt(max(0.0, var_y))


def compute_elastic_distance_ou(
    prices: Sequence[float],
) -> float:
    """Calcula a distancia elastica adimensional zeta_t do preco contra o equilibrio OU."""
    n = len(prices)
    if n < 5:
        return 0.0
    last_price = float(prices[-1])
    theta, mu, sigma = estimate_ornstein_uhlenbeck_parameters(prices)
    if theta > 1e-6 and sigma > 1e-12:
        sigma_eq = sigma / math.sqrt(2.0 * theta)
        if sigma_eq > 1e-12:
            return (last_price - mu) / sigma_eq
    sigma_std = sigma if sigma > 1e-12 else 0.0
    if sigma_std < 1e-12:
        mean_p = sum(float(p) for p in prices) / n
        var_p = sum((float(p) - mean_p) ** 2 for p in prices) / n
        sigma_std = math.sqrt(max(0.0, var_p))
    if sigma_std <= 1e-12:
        return 0.0
    return (last_price - mu) / sigma_std
