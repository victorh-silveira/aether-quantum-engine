"""Janela movel de media harmonica de bb_width para D-SQUEEZE adaptativo."""

from __future__ import annotations

from collections import deque

from src.application.services.deep_learning.dl_indicator_config import load_indicator_config_from_settings


_STATE: dict[str, deque[float] | int | None] = {"buffer": None, "maxlen": None}


def reset_bb_width_buffer() -> None:
    """Limpa buffer de bb_width para testes e reinicializacao de sessao."""
    _STATE["buffer"] = None
    _STATE["maxlen"] = None


def _buffer(maxlen: int | None = None) -> deque[float]:
    """Garante deque com maxlen de indicators.windows.bb_width_harmonic_window."""
    target = (
        int(maxlen)
        if maxlen is not None
        else int(load_indicator_config_from_settings()["windows"]["bb_width_harmonic_window"])
    )
    current = _STATE["buffer"]
    if not isinstance(current, deque) or _STATE["maxlen"] != target:
        old = list(current) if isinstance(current, deque) else []
        current = deque(old[-target:], maxlen=target)
        _STATE["buffer"] = current
        _STATE["maxlen"] = target
    return current


def bb_width_buffer_snapshot() -> tuple[float, ...]:
    """Retorna copia imutavel do buffer de bb_width."""
    return tuple(_buffer())


def record_bb_width(bb_width: float, *, harmonic_window: int | None = None) -> None:
    """Registra leitura de bb_width no buffer movel."""
    value = float(bb_width)
    if value > 0.0:
        _buffer(harmonic_window).append(value)


def harmonic_mean_bb_width(*, history: list[float] | None = None) -> float:
    """Calcula media harmonica de bb_width sobre historico movel."""
    values = list(history if history is not None else _buffer())
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
    anomaly_ratio: float,
) -> bool:
    """Indica compressao anomala quando bb_width cai abaixo do ratio da media harmonica."""
    width = float(bb_width)
    if width <= 0.0:
        return False
    mean = harmonic_mean_bb_width(history=history)
    if mean <= 0.0:
        return False
    return (width / mean) + 1e-12 < float(anomaly_ratio)


def evaluate_bb_width_squeeze(
    bb_width: float | None,
    *,
    anomaly_ratio: float,
    harmonic_window: int | None = None,
) -> tuple[bool, float, float]:
    """Registra bb_width, calcula media harmonica e retorna flag de compressao anomala."""
    if bb_width is None:
        return False, harmonic_mean_bb_width(), 0.0
    width = float(bb_width)
    record_bb_width(width, harmonic_window=harmonic_window)
    mean = harmonic_mean_bb_width()
    compressed = anomalous_bb_compression(width, anomaly_ratio=anomaly_ratio)
    return compressed, mean, width
