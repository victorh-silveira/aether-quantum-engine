"""Ajuste logístico com parada antecipada em bloco temporal interno."""

from __future__ import annotations

import math

from indicator.model import sigmoid


def fit_logistic(
    rows: list[tuple[float, ...]],
    labels: list[int],
    *,
    max_steps: int = 200,
    patience: int = 20,
    min_steps: int = 30,
    regularization: float = 0.001,
) -> tuple[list[float], float, int]:
    """Guarda os melhores pesos pelo Brier no fim temporal do treino."""
    split = max(1, min(len(labels) - 1, int(len(labels) * 0.8)))
    fit_rows, stop_rows = rows[:split], rows[split:]
    fit_labels, stop_labels = labels[:split], labels[split:]
    dimension = len(rows[0])
    weights = [0.0] * dimension
    bias = 0.0
    best_weights = list(weights)
    best_bias = bias
    best_brier = math.inf
    stale = 0
    steps = 0
    for step in range(1, max_steps + 1):
        gradient = [0.0] * dimension
        bias_gradient = 0.0
        for row, label in zip(fit_rows, fit_labels, strict=True):
            residual = sigmoid(bias + sum(weight * value for weight, value in zip(weights, row, strict=True))) - label
            bias_gradient += residual
            for index, value in enumerate(row):
                gradient[index] += residual * value
        bias -= 0.1 * bias_gradient / len(fit_labels)
        weights = [
            weight - 0.1 * (gradient[index] / len(fit_labels) + regularization * weight)
            for index, weight in enumerate(weights)
        ]
        brier = sum(
            (sigmoid(bias + sum(weight * value for weight, value in zip(weights, row, strict=True))) - label) ** 2
            for row, label in zip(stop_rows, stop_labels, strict=True)
        ) / len(stop_labels)
        steps = step
        if brier < best_brier - 1e-7:
            best_brier = brier
            best_weights = list(weights)
            best_bias = bias
            stale = 0
        else:
            stale += 1
            if step >= min_steps and stale >= patience:
                break
    return best_weights, best_bias, steps
