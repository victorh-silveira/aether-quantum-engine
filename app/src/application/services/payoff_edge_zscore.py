"""Janela movel de Z-Score sobre predicted_payoff_edge para filtro adaptativo de conviccao."""

from __future__ import annotations

import math
from collections import deque


EDGE_ZSCORE_WINDOW = 30
EDGE_ZSCORE_WIN_THRESHOLD = 0.5
EDGE_ZSCORE_STD_EPS = 1e-6
EDGE_ZSCORE_TURBO_THRESHOLD = 1.5

WIN_EXPECTED = "WIN_EXPECTED"
NO_EDGE_NEUTRAL = "NO_EDGE_NEUTRAL"
LOSS_EXPECTED = "LOSS_EXPECTED"

_edge_buffer: deque[float] = deque(maxlen=EDGE_ZSCORE_WINDOW)


def reset_payoff_edge_buffer() -> None:
    """Limpa o buffer de edge para testes e reinicializacao de sessao."""
    _edge_buffer.clear()


def payoff_edge_buffer_snapshot() -> tuple[float, ...]:
    """Retorna copia imutavel do buffer de edges para diagnostico."""
    return tuple(_edge_buffer)


def sample_edge_std(values: list[float]) -> float:
    """Calcula desvio padrao amostral de uma lista de edges."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(0.0, variance))


def compute_edge_zscore(edge: float, *, history: list[float] | None = None) -> float:
    """Calcula Z-Score do edge atual contra historico movel."""
    values = list(history if history is not None else _edge_buffer)
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    std = sample_edge_std(values)
    return (float(edge) - mean) / (std + EDGE_ZSCORE_STD_EPS)


def classify_edge_expectancy(edge: float, z_edge: float) -> str:
    """Classifica expectativa de payoff com barreira estatistica movel."""
    if float(edge) <= 0.0:
        return LOSS_EXPECTED
    if float(z_edge) + 1e-12 >= EDGE_ZSCORE_WIN_THRESHOLD:
        return WIN_EXPECTED
    return NO_EDGE_NEUTRAL


def edge_zscore_neutral_regime_active(metrics: dict) -> bool:
    """Indica regime NO_EDGE_NEUTRAL com base dinamica de 0.15% da banca."""
    return str(metrics.get("edge_expectancy") or "") == NO_EDGE_NEUTRAL


def apply_payoff_edge_zscore(edge: float) -> tuple[float, str]:
    """Registra edge no buffer, calcula Z-Score e classifica expectativa."""
    value = float(edge)
    _edge_buffer.append(value)
    z_edge = compute_edge_zscore(value)
    expectancy = classify_edge_expectancy(value, z_edge)
    return z_edge, expectancy


def attach_payoff_edge_zscore_metrics(metrics: dict, edge: float) -> tuple[float, str]:
    """Anexa telemetria de Z-Score nas metricas do ciclo."""
    z_edge, expectancy = apply_payoff_edge_zscore(edge)
    metrics["edge_zscore"] = float(z_edge)
    metrics["edge_expectancy"] = str(expectancy)
    metrics["edge_neutral_regime"] = bool(edge_zscore_neutral_regime_active(metrics))
    return z_edge, expectancy
