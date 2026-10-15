"""Metricas de validacao temporal e incerteza por blocos."""

from __future__ import annotations

import hashlib


def brier_predictions(predictions: list[float], labels: list[int]) -> float:
    """Calcula erro quadratico probabilistico medio."""
    return sum((value - label) ** 2 for value, label in zip(predictions, labels, strict=True)) / len(labels)


def expected_calibration_error(predictions: list[float], labels: list[int], bins: int = 10) -> float:
    """Mede distancia entre probabilidade media e frequencia por faixa."""
    total = len(labels)
    error = 0.0
    for index in range(bins):
        pairs = [
            (probability, label)
            for probability, label in zip(predictions, labels, strict=True)
            if min(bins - 1, int(probability * bins)) == index
        ]
        if pairs:
            error += (
                abs(
                    sum(probability for probability, _ in pairs) / len(pairs)
                    - sum(label for _, label in pairs) / len(pairs)
                )
                * len(pairs)
                / total
            )
    return error


def block_bootstrap_lower(improvements: list[float], block: int = 12, repetitions: int = 200) -> float:
    """Limite inferior unilateral empirico com reamostragem de blocos contiguos."""
    if not improvements:
        raise ValueError("amostra vazia")
    size = len(improvements)
    width = min(block, size)
    means = []
    draw = 0
    for _ in range(repetitions):
        sampled = []
        while len(sampled) < size:
            digest = hashlib.sha256(f"20261013:{draw}".encode()).digest()
            draw += 1
            start = int.from_bytes(digest[:8], "big") % (size - width + 1)
            sampled.extend(improvements[start : start + width])
        means.append(sum(sampled[:size]) / size)
    means.sort()
    return means[max(0, int(repetitions * 0.05) - 1)]


def momentum_reference(rows: list[tuple[float, ...]]) -> list[float]:
    """Referencia fixa da direcao do retorno mais recente."""
    return [0.55 if row[-1] > 0 else 0.45 if row[-1] < 0 else 0.5 for row in rows]
