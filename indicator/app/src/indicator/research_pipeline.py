"""Pesquisa sequencial com selecao walk-forward e teste intocado."""

from __future__ import annotations

import math
from dataclasses import dataclass

from indicator.model import LOOKBACK, sigmoid
from indicator.optimizer import fit_logistic
from indicator.research import calibrate
from indicator.validation import (
    block_bootstrap_lower,
    brier_predictions,
    expected_calibration_error,
    momentum_reference,
)

SPECS = (("close", 0.001), ("close", 0.05), ("ohlc", 0.001), ("ohlc", 0.05), ("tree", 0.0))
EMBARGO = LOOKBACK + 1


@dataclass(frozen=True)
class Fitted:
    """Parametros ajustados somente no passado da janela avaliada."""

    feature_set: str
    regularization: float
    weights: tuple[float, ...]
    bias: float
    means: tuple[float, ...]
    scales: tuple[float, ...]
    steps: int
    tree_feature: int = -1
    tree_threshold: float = 0.0
    tree_left: float = 0.5
    tree_right: float = 0.5

    def score(self, row: tuple[float, ...]) -> float:
        """Produz logit sem acessar labels posteriores."""
        if self.feature_set == "tree":
            probability = self.tree_left if row[self.tree_feature] <= self.tree_threshold else self.tree_right
            return math.log(probability / (1.0 - probability))
        return self.bias + sum(
            weight * (value - mean) / scale
            for weight, value, mean, scale in zip(self.weights, row, self.means, self.scales, strict=True)
        )


@dataclass(frozen=True)
class ResearchResult:
    """Modelo escolhido e diagnostico no bloco de teste intocado."""

    fitted: Fitted
    slope: float
    offset: float
    accuracy: float
    baseline_accuracy: float
    brier: float
    baseline_brier: float
    ece: float
    improvement_lower95: float
    predictions: tuple[float, ...]
    validation_samples: int
    candidate_briers: tuple[tuple[float, float], ...]
    probability_quantiles: tuple[float, float, float]
    regime_briers: tuple[tuple[str, int, float], ...]


def fit_candidate(rows: list[tuple[float, ...]], labels: list[int], spec: tuple[str, float]) -> Fitted:
    """Ajusta uma configuracao sem trocar modelo em producao."""
    feature_set, regularization = spec
    dimension = len(rows[0])
    if feature_set == "tree":
        feature = LOOKBACK * 2 - 1
        left = [label for row, label in zip(rows, labels, strict=True) if row[feature] <= 0.0]
        right = [label for row, label in zip(rows, labels, strict=True) if row[feature] > 0.0]
        return Fitted(
            feature_set,
            0.0,
            (0.0,) * dimension,
            0.0,
            (0.0,) * dimension,
            (1.0,) * dimension,
            1,
            feature,
            0.0,
            (sum(left) + 1) / (len(left) + 2),
            (sum(right) + 1) / (len(right) + 2),
        )
    means = tuple(sum(row[index] for row in rows) / len(rows) for index in range(dimension))
    scales = tuple(
        max((sum((row[index] - means[index]) ** 2 for row in rows) / len(rows)) ** 0.5, 1e-8)
        for index in range(dimension)
    )
    normalized = [
        tuple((value - mean) / scale for value, mean, scale in zip(row, means, scales, strict=True)) for row in rows
    ]
    weights, bias, steps = fit_logistic(
        normalized, labels, max_steps=40, patience=10, min_steps=20, regularization=regularization
    )
    return Fitted(feature_set, regularization, tuple(weights), bias, means, scales, steps)


def run_research(
    close_rows: list[tuple[float, ...]],
    ohlc_rows: list[tuple[float, ...]],
    labels: list[int],
) -> ResearchResult:
    """Escolhe entre cinco configuracoes em tres janelas e testa 20% finais."""
    total = len(labels)
    test_start = int(total * 0.8)
    development_end = test_start - EMBARGO
    calibration_start = int(development_end * 0.7)
    rows_by_set = {"close": close_rows, "ohlc": ohlc_rows, "tree": ohlc_rows}
    scores = []
    for spec in SPECS:
        rows = rows_by_set[spec[0]]
        fold_scores = []
        for fraction in (0.3, 0.45, 0.6):
            train_end = int(development_end * fraction)
            validation_start = train_end + EMBARGO
            validation_end = min(int(development_end * (fraction + 0.1)), calibration_start)
            fitted = fit_candidate(rows[:train_end], labels[:train_end], spec)
            predictions = [sigmoid(fitted.score(row)) for row in rows[validation_start:validation_end]]
            fold_scores.append(brier_predictions(predictions, labels[validation_start:validation_end]))
        scores.append((sum(fold_scores) / len(fold_scores), spec))
    _, selected = min(scores, key=lambda item: item[0])
    rows = rows_by_set[selected[0]]
    fitted = fit_candidate(rows[:calibration_start], labels[:calibration_start], selected)
    calibration_rows = rows[calibration_start + EMBARGO : development_end]
    calibration_labels = labels[calibration_start + EMBARGO : development_end]
    slope, offset = calibrate([fitted.score(row) for row in calibration_rows], calibration_labels)
    test_rows = rows[test_start:]
    test_labels = labels[test_start:]
    predictions = [sigmoid(slope * fitted.score(row) + offset) for row in test_rows]
    majority = sum(labels[:calibration_start]) / calibration_start
    references = ([majority] * len(test_labels), momentum_reference(close_rows[test_start:]))
    reference = min(references, key=lambda item: brier_predictions(item, test_labels))
    brier = brier_predictions(predictions, test_labels)
    baseline_brier = brier_predictions(reference, test_labels)
    differences = [
        (reference_p - label) ** 2 - (candidate_p - label) ** 2
        for candidate_p, reference_p, label in zip(predictions, reference, test_labels, strict=True)
    ]
    sorted_predictions = sorted(predictions)
    quantiles = tuple(sorted_predictions[int((len(sorted_predictions) - 1) * fraction)] for fraction in (0.1, 0.5, 0.9))
    threshold = sorted(row[-1] for row in ohlc_rows[:calibration_start])[calibration_start // 2]
    regimes = []
    for name, high in (("baixa", False), ("alta", True)):
        pairs = [
            (prediction, label)
            for row, prediction, label in zip(ohlc_rows[test_start:], predictions, test_labels, strict=True)
            if (row[-1] > threshold) == high
        ]
        if pairs:
            regimes.append(
                (
                    name,
                    len(pairs),
                    brier_predictions([prediction for prediction, _ in pairs], [label for _, label in pairs]),
                )
            )
    return ResearchResult(
        fitted,
        slope,
        offset,
        sum((value >= 0.5) == bool(label) for value, label in zip(predictions, test_labels, strict=True))
        / len(test_labels),
        sum((value >= 0.5) == bool(label) for value, label in zip(reference, test_labels, strict=True))
        / len(test_labels),
        brier,
        baseline_brier,
        expected_calibration_error(predictions, test_labels),
        block_bootstrap_lower(differences),
        tuple(predictions),
        len(test_labels),
        tuple((spec[1], score) for score, spec in scores),
        quantiles,
        tuple(regimes),
    )
