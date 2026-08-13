"""Escala temporal de knobs em barras para sweep multi-TF (ancora M2)."""

from __future__ import annotations

from typing import Any


ANCHOR_MICRO_SECONDS = 120
LOOKBACK_ANCHOR_BARS = 720
HISTORY_ANCHOR_BARS = 2000
LABEL_MA_ANCHOR = 8
LABEL_SMOOTH_ANCHOR = 2
IMPLIED_VOL_ANCHOR = 120
MINI_BARS_ANCHOR = 120
SLOPE_BARS_ANCHOR = 5
BASELINE_LOOKBACK_ANCHOR = 72
LOOKBACK_MIN = 96
LOOKBACK_MAX = 1440
HISTORY_MAX = 4000
HISTORY_MIN_ABS = 800


def scale_bars(bars: int, micro_seconds: int, *, anchor: int = ANCHOR_MICRO_SECONDS) -> int:
    """Converte barras da ancora para o micro alvo (wall-clock constante)."""
    micro = max(1, int(micro_seconds))
    base = max(1, int(bars))
    scaled = int(round(base * float(anchor) / float(micro)))
    return max(1, scaled)


def scale_lookback(micro_seconds: int) -> int:
    """Lookback em barras com piso/teto para TCN."""
    raw = scale_bars(LOOKBACK_ANCHOR_BARS, micro_seconds)
    return max(LOOKBACK_MIN, min(LOOKBACK_MAX, raw))


def scale_history(micro_seconds: int, lookback: int) -> int:
    """Historico >= lookback + margem, com piso absoluto e teto."""
    raw = scale_bars(HISTORY_ANCHOR_BARS, micro_seconds)
    need = int(lookback) + 400
    return max(HISTORY_MIN_ABS, need, min(HISTORY_MAX, max(raw, need)))


def _scale_int_map(block: dict[str, Any], micro_seconds: int) -> dict[str, Any]:
    """Escala valores inteiros (e floats integrais) de um mapa de barras."""
    out: dict[str, Any] = {}
    for key, value in block.items():
        if isinstance(value, bool):
            out[key] = value
        elif isinstance(value, int):
            out[key] = scale_bars(value, micro_seconds)
        elif isinstance(value, float) and value.is_integer():
            out[key] = float(scale_bars(int(value), micro_seconds))
        else:
            out[key] = value
    return out


def apply_tf_wallclock_scale(settings: dict[str, Any], micro_seconds: int) -> dict[str, Any]:
    """Ajusta lookback/history/label/indicators/live bars para o micro alvo."""
    micro = max(1, int(micro_seconds))
    lookback = scale_lookback(micro)
    history = scale_history(micro, lookback)
    dl = settings.setdefault("deep_learning", {})
    if not isinstance(dl, dict):
        raise ValueError("deep_learning invalido")
    dl["lookback"] = lookback
    dl["training_history_bars"] = history
    dl["label_ma_window"] = max(2, scale_bars(LABEL_MA_ANCHOR, micro))
    dl["label_smooth_bars"] = max(1, scale_bars(LABEL_SMOOTH_ANCHOR, micro))
    dl["implied_vol_bars"] = max(8, scale_bars(IMPLIED_VOL_ANCHOR, micro))
    indicators = dl.get("indicators")
    if isinstance(indicators, dict):
        windows = indicators.get("windows")
        if isinstance(windows, dict):
            indicators["windows"] = _scale_int_map(windows, micro)
        for nest_key in ("congestion", "trend_consensus"):
            nest = indicators.get(nest_key)
            if isinstance(nest, dict) and "min_bars" in nest:
                nest["min_bars"] = max(8, scale_bars(int(nest["min_bars"]), micro))
    gate = dl.get("deploy_gate")
    if isinstance(gate, dict) and "mini_bars" in gate:
        gate["mini_bars"] = max(16, scale_bars(MINI_BARS_ANCHOR, micro))
        gate["max_eval_steps"] = max(int(gate.get("max_eval_steps", 24) or 24), 48)
        gate["min_trades"] = max(int(gate.get("min_trades", 2) or 2), 16)
    data = settings.setdefault("data_handler", {})
    if isinstance(data, dict):
        data["micro_history_bars"] = history
        data["micro_fetch_count"] = history
        data["fetch_count"] = history
    orch = settings.get("orchestrator")
    if isinstance(orch, dict):
        execution = orch.get("execution")
        if isinstance(execution, dict):
            scale_vision = execution.get("scale_vision")
            if isinstance(scale_vision, dict) and "slope_bars" in scale_vision:
                scale_vision["slope_bars"] = max(2, scale_bars(SLOPE_BARS_ANCHOR, micro))
            dyn = execution.get("dynamic_threshold")
            if isinstance(dyn, dict) and "baseline_lookback" in dyn:
                dyn["baseline_lookback"] = max(12, scale_bars(BASELINE_LOOKBACK_ANCHOR, micro))
    return settings
