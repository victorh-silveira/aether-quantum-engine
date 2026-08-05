"""Classificador de regime micro: explosao, retracao ou chop (sem SKIP)."""

from __future__ import annotations

from typing import Any

from src.domain.models.trade import TradeDirection


_VALID = {TradeDirection.CALL.name, TradeDirection.PUT.name}
_REGIME_EXPLOSION = "explosion"
_REGIME_RETRACTION = "retraction"
_REGIME_CHOP = "chop"


def _side(value: object) -> str | None:
    """Normaliza CALL/PUT ou None."""
    side = str(value or "").strip().upper()
    return side if side in _VALID else None


def _tick_confirms_side(metrics: dict[str, Any], side: str) -> bool:
    """True se velocity/acceleration de ticks apoia o lado vivo."""
    flow = metrics.get("flow_features") if isinstance(metrics.get("flow_features"), dict) else {}
    vel = 0.0
    accel = 0.0
    try:
        vel = float(flow.get("price_velocity") or flow.get("micro_tick_velocity") or 0.0)
    except (TypeError, ValueError):
        vel = 0.0
    try:
        accel = float(flow.get("micro_tick_acceleration") or flow.get("price_acceleration") or 0.0)
    except (TypeError, ValueError):
        accel = 0.0
    score = vel + 0.5 * accel
    if abs(score) <= 1e-12:
        return False
    tick_side = TradeDirection.CALL.name if score > 0.0 else TradeDirection.PUT.name
    return tick_side == side


def classify_micro_regime(
    metrics: dict[str, Any],
    tcn_dir: str | None,
    *,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classifica explosion/retraction/chop e marca retracao contra o TCN."""
    vision = cfg if isinstance(cfg, dict) else {}
    require_mili = bool(vision.get("retraction_require_mili", True))
    use_tick = bool(vision.get("retraction_use_tick_accel", True))
    tcn = _side(tcn_dir) or _side(metrics.get("tcn_direction")) or _side(metrics.get("scale_micro_dir"))
    mi_prev = _side(metrics.get("scale_mini_prev_bar_dir"))
    mi_curr = _side(metrics.get("scale_mini_bar_dir"))
    mili = _side(metrics.get("scale_mili_dir"))
    metrics.setdefault("scale_micro_regime", _REGIME_CHOP)
    metrics.setdefault("scale_micro_side", None)
    metrics.setdefault("scale_retraction_vs_tcn", False)
    metrics.setdefault("scale_mili_oppose_tcn", False)
    if tcn is not None and mili is not None and mili != tcn:
        metrics["scale_mili_oppose_tcn"] = True
    live = mi_curr if mi_curr is not None else mili
    metrics["scale_micro_side"] = live
    if mi_prev is not None and mi_curr is not None and mi_prev == mi_curr:
        if mili is None or mili == mi_curr:
            metrics["scale_micro_regime"] = _REGIME_EXPLOSION
            metrics["scale_micro_side"] = mi_curr
            metrics["scale_retraction_vs_tcn"] = False
            return metrics
        metrics["scale_micro_regime"] = _REGIME_CHOP
        metrics["scale_retraction_vs_tcn"] = False
        return metrics
    if mi_prev is not None and mi_curr is not None and mi_prev != mi_curr:
        mili_ok = mili == mi_curr
        tick_ok = use_tick and _tick_confirms_side(metrics, mi_curr)
        confirmed = mili_ok if require_mili else bool(mili_ok or tick_ok)
        if confirmed:
            metrics["scale_micro_regime"] = _REGIME_RETRACTION
            metrics["scale_micro_side"] = mi_curr
            metrics["scale_retraction_vs_tcn"] = bool(tcn is not None and mi_curr != tcn)
            return metrics
    metrics["scale_micro_regime"] = _REGIME_CHOP
    metrics["scale_retraction_vs_tcn"] = False
    return metrics


def micro_regime_token(regime: object) -> str:
    """Abrevia regime para token IND."""
    name = str(regime or "").strip().lower()
    if name == _REGIME_EXPLOSION:
        return "explos"
    if name == _REGIME_RETRACTION:
        return "retract"
    return "chop"
