"""Janela movel adaptativa de Z-Score sobre predicted_payoff_edge."""

from __future__ import annotations

import math
from collections import deque
from typing import Any


EDGE_ZSCORE_WINDOW_MIN = 15
EDGE_ZSCORE_WINDOW_DEFAULT = 30
EDGE_ZSCORE_WINDOW_MAX = 45
EDGE_ZSCORE_WIN_THRESHOLD = 0.5
EDGE_ZSCORE_STD_EPS = 1e-6
EDGE_ZSCORE_TURBO_THRESHOLD = 1.5
HURST_TREND_FLOOR = 0.55
ATR_TRANSITION_SCALE = 0.15
BB_COMPRESSION_WIDTH = 0.12

_edge_buffers: dict[str, deque[float]] = {}


def _buffer_key(symbol: str | None) -> str:
    """Normaliza chave de buffer por simbolo."""
    return str(symbol or "_global")


def _edge_buffer_for(symbol: str | None = None) -> deque[float]:
    """Retorna buffer de edge isolado por simbolo."""
    key = _buffer_key(symbol)
    buf = _edge_buffers.get(key)
    if buf is None:
        buf = deque(maxlen=EDGE_ZSCORE_WINDOW_MAX)
        _edge_buffers[key] = buf
    return buf


def reset_payoff_edge_buffer(symbol: str | None = None) -> None:
    """Limpa buffer de edge para testes e reinicializacao de sessao."""
    if symbol is None:
        _edge_buffers.clear()
        return
    _edge_buffers.pop(_buffer_key(symbol), None)


def payoff_edge_buffer_snapshot(symbol: str | None = None) -> tuple[float, ...]:
    """Retorna copia imutavel do buffer de edges para diagnostico."""
    return tuple(_edge_buffer_for(symbol))


def _indicator_map(metrics: dict[str, Any] | None) -> dict[str, Any]:
    """Extrai os indicadores técnicos dos metadados de entrada."""
    if not isinstance(metrics, dict):
        return {}
    chunk = metrics.get("indicators")
    if isinstance(chunk, dict):
        return chunk
    micro = metrics.get("micro_indicators")
    if isinstance(micro, dict):
        return micro
    return {}


def _hurst_transition_factor(indicators: dict[str, Any]) -> float:
    """Calcula o fator de transicao do Hurst Exponent."""
    hurst = float(indicators.get("hurst", 0.5) or 0.5)
    return max(0.0, min(1.0, (hurst - HURST_TREND_FLOOR) / max(1e-6, 0.70 - HURST_TREND_FLOOR)))


def _atr_transition_factor(metrics: dict[str, Any], indicators: dict[str, Any]) -> float:
    """Calcula o fator de transicao baseado na razao de mudanca do ATR."""
    raw_change = metrics.get("atr_change_ratio")
    if raw_change is None:
        raw_change = indicators.get("atr_change_ratio")
    if raw_change is None:
        atr_norm = float(indicators.get("atr_norm", indicators.get("atr", 0.0)) or 0.0)
        raw_change = atr_norm
    return max(0.0, min(1.0, abs(float(raw_change or 0.0)) / ATR_TRANSITION_SCALE))


def _compression_expansion_factor(indicators: dict[str, Any]) -> float:
    """Calcula o fator de compressao/expansao das Bandas de Bollinger."""
    bb_width = float(indicators.get("bb_width", 0.0) or 0.0)
    if bb_width <= 0.0:
        return 0.0
    compression = 1.0 - min(1.0, bb_width / BB_COMPRESSION_WIDTH)
    return max(0.0, min(1.0, compression))


def resolve_adaptive_edge_window(metrics: dict[str, Any] | None = None) -> int:
    """Encolhe a janela em tendencia/vol alta e expande em lateral ruidoso."""
    if not isinstance(metrics, dict):
        return EDGE_ZSCORE_WINDOW_DEFAULT
    indicators = _indicator_map(metrics)
    hurst_factor = _hurst_transition_factor(indicators)
    atr_factor = _atr_transition_factor(metrics, indicators)
    compression = _compression_expansion_factor(indicators)
    transition = max(hurst_factor, atr_factor) * (1.0 - 0.35 * compression)
    span = EDGE_ZSCORE_WINDOW_MAX - EDGE_ZSCORE_WINDOW_MIN
    window = int(round(EDGE_ZSCORE_WINDOW_MAX - transition * span))
    return max(EDGE_ZSCORE_WINDOW_MIN, min(EDGE_ZSCORE_WINDOW_MAX, window))


def _active_edge_history(metrics: dict[str, Any] | None = None, *, symbol: str | None = None) -> list[float]:
    """Retorna o historico de edges limitado a janela adaptativa atual."""
    window = resolve_adaptive_edge_window(metrics)
    values = list(_edge_buffer_for(symbol))
    if len(values) <= window:
        return values
    return values[-window:]


def sample_edge_std(values: list[float]) -> float:
    """Calcula desvio padrao amostral de uma lista de edges."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(0.0, variance))


def compute_edge_zscore(
    edge: float,
    *,
    history: list[float] | None = None,
    metrics: dict[str, Any] | None = None,
    symbol: str | None = None,
) -> float:
    """Calcula Z-Score do edge atual contra historico movel adaptativo."""
    values = list(history) if history is not None else _active_edge_history(metrics, symbol=symbol)
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    std = sample_edge_std(values)
    return (float(edge) - mean) / (std + EDGE_ZSCORE_STD_EPS)


def apply_payoff_edge_zscore(
    edge: float,
    metrics: dict[str, Any] | None = None,
    *,
    symbol: str | None = None,
) -> float:
    """Registra edge no buffer do simbolo e calcula Z-Score adaptativo."""
    value = float(edge)
    buf = _edge_buffer_for(symbol)
    buf.append(value)
    window = resolve_adaptive_edge_window(metrics)
    z_edge = compute_edge_zscore(value, metrics=metrics, symbol=symbol)
    if isinstance(metrics, dict):
        metrics["edge_zscore_window"] = int(window)
    return z_edge


def attach_payoff_edge_zscore_metrics(metrics: dict, edge: float, *, symbol: str | None = None) -> float:
    """Anexa telemetria de Z-Score adaptativo nas metricas do ciclo."""
    resolved_symbol = symbol or str(metrics.get("symbol") or "")
    z_edge = apply_payoff_edge_zscore(edge, metrics, symbol=resolved_symbol or None)
    metrics["edge_zscore"] = float(z_edge)
    metrics["meta_payoff_edge_zscore"] = float(z_edge)
    metrics["edge_zscore_samples"] = len(_edge_buffer_for(resolved_symbol or None))
    return z_edge
