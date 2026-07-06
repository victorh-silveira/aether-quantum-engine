"""Janela movel de media harmonica de bb_width para D-SQUEEZE adaptativo."""

from __future__ import annotations

from collections import deque


BB_WIDTH_HARMONIC_WINDOW = 50
BB_WIDTH_ANOMALY_RATIO = 0.55

_bb_width_buffer: deque[float] = deque(maxlen=BB_WIDTH_HARMONIC_WINDOW)


def reset_bb_width_buffer() -> None:
    """Limpa buffer de bb_width para testes e reinicializacao de sessao."""
    _bb_width_buffer.clear()


def bb_width_buffer_snapshot() -> tuple[float, ...]:
    """Retorna copia imutavel do buffer de bb_width."""
    return tuple(_bb_width_buffer)


def record_bb_width(bb_width: float) -> None:
    """Registra leitura de bb_width no buffer movel."""
    value = float(bb_width)
    if value > 0.0:
        _bb_width_buffer.append(value)


def harmonic_mean_bb_width(*, history: list[float] | None = None) -> float:
    """Calcula media harmonica de bb_width sobre historico movel."""
    values = list(history if history is not None else _bb_width_buffer)
    positives = [float(value) for value in values if float(value) > 0.0]
    if not positives:
        return 0.0
    reciprocal_sum = sum(1.0 / value for value in positives)
    if reciprocal_sum <= 0.0:
        return 0.0
    return len(positives) / reciprocal_sum


def anomalous_bb_compression(
    bb_width: float,
    *,
    history: list[float] | None = None,
    anomaly_ratio: float | None = None,
) -> bool:
    """Indica compressao anomala quando bb_width cai abaixo do ratio da media harmonica."""
    width = float(bb_width)
    if width <= 0.0:
        return False
    mean = harmonic_mean_bb_width(history=history)
    if mean <= 0.0:
        return False
    ratio = BB_WIDTH_ANOMALY_RATIO if anomaly_ratio is None else float(anomaly_ratio)
    return (width / mean) + 1e-12 < ratio


def evaluate_bb_width_squeeze(
    bb_width: float | None,
    *,
    anomaly_ratio: float | None = None,
) -> tuple[bool, float, float]:
    """Registra bb_width, calcula media harmonica e retorna flag de compressao anomala."""
    if bb_width is None:
        return False, harmonic_mean_bb_width(), 0.0
    width = float(bb_width)
    record_bb_width(width)
    mean = harmonic_mean_bb_width()
    compressed = anomalous_bb_compression(width, anomaly_ratio=anomaly_ratio)
    return compressed, mean, width
