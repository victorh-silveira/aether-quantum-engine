"""Parametros de cotacao final para contratos Rise/Fall."""

from __future__ import annotations

from typing import Any

from src.application.services.rise_fall_quote_guard import resolve_effective_side_probability


def _attach_quote_guard_params(
    params: dict, metrics: dict | None, direction: Any, exec_cfg: dict, *, trading_mode: str = "real"
) -> None:
    """Anexa probabilidade efetiva e piso de edge para quote guard no proposal."""
    if not isinstance(metrics, dict) or not bool(exec_cfg.get("require_quote_edge", False)):
        params.pop("_quote_guard_side_probability", None)
        params.pop("_quote_guard_min_edge", None)
        params.pop("_quote_guard_safety_margin", None)
        params.pop("min_payout_rate", None)
        return
    dir_name = getattr(direction, "name", str(direction)).upper()
    p_side = resolve_effective_side_probability(metrics, dir_name)
    if p_side is None:
        return
    if metrics.get("indicator_demo_pilot") and trading_mode.lower() == "demo":
        setup = str(metrics.get("indicator_model_version", "")).startswith("pullback-1-")
        params["_indicator_demo_pilot"] = True
        params["_indicator_direct_buy_fallback"] = not setup
        params["_indicator_candle_epoch"] = int(metrics.get("indicator_candle_epoch") or 0)
        params["min_payout_rate"] = 0.0
        if setup:
            params.pop("_quote_guard_side_probability", None)
            params.pop("_quote_guard_min_edge", None)
            params.pop("_quote_guard_safety_margin", None)
            return
        haircut = float(exec_cfg.get("quote_probability_haircut", 0.0) or 0.0)
        params["_quote_guard_side_probability"] = max(0.0, p_side - haircut)
        params["_quote_guard_min_edge"] = -1.0
        params["_quote_guard_safety_margin"] = 0.0
        return
    try:
        haircut = float(exec_cfg.get("quote_probability_haircut", 0.0) or 0.0)
        p_eff = max(0.0, p_side - haircut)
        is_rec = float(metrics.get("pending_loss_total", 0.0) or 0.0) > 0.5
        raw_floor = exec_cfg.get("recovery_neg_edge_floor", 0.0) if is_rec else exec_cfg.get("min_edge_execute", 0.0)
        min_edge = max(0.0, float(raw_floor or 0.0))
        params["_quote_guard_side_probability"] = p_eff
        params["_quote_guard_min_edge"] = min_edge
        params["_quote_guard_safety_margin"] = float(exec_cfg.get("quote_safety_margin", 0.0) or 0.0)
        params["min_payout_rate"] = float(exec_cfg.get("min_payout_rate", 0.76) or 0.0)
    except (TypeError, ValueError):
        pass
