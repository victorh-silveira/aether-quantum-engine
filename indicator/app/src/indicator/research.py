"""Selecao temporal e calibracao conservadora do indicador."""

from __future__ import annotations

from indicator.model import sigmoid
from indicator.optimizer import fit_logistic

REGULARIZATIONS = (0.001, 0.05)


def logits(rows: list[tuple[float, ...]], weights: list[float], bias: float) -> list[float]:
    """Calcula logits sem usar labels futuros."""
    return [bias + sum(weight * value for weight, value in zip(weights, row, strict=True)) for row in rows]


def brier(scores: list[float], labels: list[int], slope: float = 1.0, offset: float = 0.0) -> float:
    """Mede o erro probabilistico no bloco cronologico recebido."""
    return sum(
        (sigmoid(slope * score + offset) - label) ** 2 for score, label in zip(scores, labels, strict=True)
    ) / len(labels)


def calibrate(scores: list[float], labels: list[int]) -> tuple[float, float]:
    """Ajusta somente shrinkage e intercepto no bloco de calibracao."""
    slope = 1.0
    offset = 0.0
    initial = brier(scores, labels)
    for _ in range(300):
        residuals = [sigmoid(slope * score + offset) - label for score, label in zip(scores, labels, strict=True)]
        offset -= 0.1 * sum(residuals) / len(labels)
        slope -= 0.1 * sum(residual * score for residual, score in zip(residuals, scores, strict=True)) / len(labels)
        slope = min(5.0, max(0.0, slope))
    return (slope, offset) if brier(scores, labels, slope, offset) < initial else (1.0, 0.0)


def select_and_calibrate(
    fit_rows: list[tuple[float, ...]],
    fit_labels: list[int],
    select_rows: list[tuple[float, ...]],
    select_labels: list[int],
    calibration_rows: list[tuple[float, ...]],
    calibration_labels: list[int],
) -> tuple[list[float], float, int, float, float, float, tuple[tuple[float, float], ...]]:
    """Seleciona hiperparametro antes de calibrar e validar no tempo."""
    candidates = []
    for regularization in REGULARIZATIONS:
        weights, bias, steps = fit_logistic(fit_rows, fit_labels, max_steps=80, regularization=regularization)
        score = brier(logits(select_rows, weights, bias), select_labels)
        candidates.append((score, regularization, weights, bias, steps))
    winner = min(candidates, key=lambda item: (item[0], item[1]))
    _, regularization, weights, bias, steps = winner
    slope, offset = calibrate(logits(calibration_rows, weights, bias), calibration_labels)
    scores = tuple((reg, score) for score, reg, _, _, _ in candidates)
    return weights, bias, steps, regularization, slope, offset, scores
