"""Salvaguardas de confluencia de mercado: exaustao, chop e contratendencia."""

from __future__ import annotations

from typing import Any

from src.application.services.execution_signal_skips import (
    _closed_candle_dir,
    _mark_skip,
)
from src.domain.models.trade import TradeDirection


_VALID = {TradeDirection.CALL.name, TradeDirection.PUT.name}


def _extract_indicator_float(metrics: dict[str, Any], key: str) -> float | None:
    """Extrai float de indicators ou metrics raiz."""
    ind = metrics.get("indicators")
    val = ind.get(key) if isinstance(ind, dict) else None
    if val is None:
        val = metrics.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _extract_edge_float(metrics: dict[str, Any]) -> float:
    """Extrai edge continuo das metricas com fallback 0.0."""
    raw = metrics.get("cal_side_edge", metrics.get("edge", 0.0))
    try:
        return float(raw or 0.0)
    except (TypeError, ValueError):
        return 0.0


def should_skip_exhaustion(
    metrics: dict[str, Any],
    exec_dir: TradeDirection,
    exec_cfg: dict[str, Any] | None,
    *,
    force: bool = False,
) -> bool:
    """Bloqueia ordens disparadas em niveis extremos de sobrecompra/sobrevenda."""
    if force or not bool((exec_cfg or {}).get("skip_exhaustion", False)):
        return False
    rsi = _extract_indicator_float(metrics, "rsi")
    bb_b = _extract_indicator_float(metrics, "bb_pct_b")
    if rsi is None or bb_b is None:
        return False
    edge = _extract_edge_float(metrics)
    if exec_dir == TradeDirection.CALL and rsi > 0.75 and bb_b > 1.05 and edge < 0.080:
        _mark_skip(metrics, "exhaustion_call", rsi=float(rsi), bb_pct_b=float(bb_b))
        return True
    if exec_dir == TradeDirection.PUT and rsi < 0.25 and bb_b < -0.05 and edge < 0.080:
        _mark_skip(metrics, "exhaustion_put", rsi=float(rsi), bb_pct_b=float(bb_b))
        return True
    return False


def should_skip_chop_congestion(
    metrics: dict[str, Any],
    exec_cfg: dict[str, Any] | None,
    *,
    force: bool = False,
) -> bool:
    """Bloqueia operacoes em consolidacao extrema, baixa volatilidade e ausencia de tendencia."""
    if force or not bool((exec_cfg or {}).get("skip_chop_congestion", False)):
        return False
    adx = _extract_indicator_float(metrics, "adx")
    bb_w = _extract_indicator_float(metrics, "bb_width_raw")
    if bb_w is None:
        bb_w = _extract_indicator_float(metrics, "bb_width")
    if adx is None:
        return False
    edge = _extract_edge_float(metrics)
    vol_ratio = _extract_indicator_float(metrics, "vol_ratio") or 1.0
    if adx < 0.20 and not (vol_ratio >= 1.35 and edge >= 0.20):
        _mark_skip(metrics, "chop_congestion", adx=float(adx), bb_width=float(bb_w or 0.0))
        return True
    if adx < 0.22 and bb_w is not None and (bb_w < 0.0 or bb_w < 0.035) and edge < 0.120:
        _mark_skip(metrics, "chop_congestion", adx=float(adx), bb_width=float(bb_w))
        return True
    return False


def should_skip_two_bar_momentum_trap(
    metrics: dict[str, Any],
    exec_dir: TradeDirection,
    exec_cfg: dict[str, Any] | None,
    *,
    force: bool = False,
) -> bool:
    """Bloqueia ordens contra duas velas M5 consecutivas sem conviccao forte."""
    if force or not bool((exec_cfg or {}).get("skip_two_bar_counter_trend", False)):
        return False
    prev_bar = str(metrics.get("scale_micro_prev_bar_dir") or "").strip().upper()
    curr_bar = str(metrics.get("scale_micro_bar_dir") or _closed_candle_dir(metrics) or "").strip().upper()
    if prev_bar not in _VALID or curr_bar not in _VALID:
        return False
    exec_name = exec_dir.name
    if exec_name not in (prev_bar, curr_bar):
        edge = _extract_edge_float(metrics)
        if edge < 0.065:
            _mark_skip(
                metrics,
                "two_bar_counter_trend",
                exec_pre_skip=exec_name,
                prev_bar=prev_bar,
                curr_bar=curr_bar,
            )
            return True
    return False


def should_skip_directional_momentum_discord(
    metrics: dict[str, Any],
    exec_dir: TradeDirection,
    exec_cfg: dict[str, Any] | None,
    *,
    force: bool = False,
) -> bool:
    """Bloqueia ordens contra pressao direcional intensa de momentum (DI e RSI)."""
    if force or not bool((exec_cfg or {}).get("skip_directional_momentum_discord", False)):
        return False
    di_diff = _extract_indicator_float(metrics, "di_diff")
    rsi = _extract_indicator_float(metrics, "rsi")
    if di_diff is None or rsi is None:
        return False
    edge = _extract_edge_float(metrics)
    if edge >= 0.080:
        return False
    if exec_dir == TradeDirection.CALL and di_diff < -0.15 and rsi < 0.45:
        _mark_skip(metrics, "bearish_momentum_discord", di_diff=float(di_diff), rsi=float(rsi))
        return True
    if exec_dir == TradeDirection.PUT and di_diff > 0.15 and rsi > 0.55:
        _mark_skip(metrics, "bullish_momentum_discord", di_diff=float(di_diff), rsi=float(rsi))
        return True
    return False
