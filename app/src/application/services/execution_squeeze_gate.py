"""Gate de consenso direcional em regime de squeeze extremo."""

from __future__ import annotations

from src.application.services.execution_squeeze_bias import normalize_bias_side
from src.domain.models.trade import TradeDirection


def squeeze_consensus_side(metrics: dict) -> str | None:
    """Infere lado dominante quando trend, regime e DL concordam."""
    trend = normalize_bias_side(metrics.get("trend_direction"))
    regime = normalize_bias_side(metrics.get("indicator_regime_side"))
    dl_side = normalize_bias_side(metrics.get("calibrated_prob"))
    sides = [s for s in (trend, regime, dl_side) if s is not None]
    if len(sides) < 2 or len(set(sides)) != 1:
        return None
    return sides[0]


def _resolved_side(metrics: dict) -> str | None:
    """Lado resolvido como call/put."""
    direction = metrics.get("resolved_direction")
    if direction is None:
        return None
    if isinstance(direction, TradeDirection):
        return "call" if direction == TradeDirection.CALL else "put"
    token = str(direction).strip().upper()
    if token in {"CALL", "RISE"}:
        return "call"
    if token in {"PUT", "FALL"}:
        return "put"
    return None


def passes_squeeze_gate(metrics: dict, *, cfg: dict | None = None) -> bool:
    """Em squeeze extremo exige margem minima e consenso sem inversao."""
    chunk = cfg if isinstance(cfg, dict) else {}
    if not bool(metrics.get("squeeze_extreme")):
        return True
    min_margin = float(chunk.get("squeeze_min_margin", 0.12))
    if float(metrics.get("direction_margin", 0.0)) + 1e-9 < min_margin:
        return False
    if metrics.get("direction_inverted") or metrics.get("low_val_flip"):
        return False
    if not bool(chunk.get("require_indicator_consensus", True)):
        return True
    consensus = squeeze_consensus_side(metrics)
    if consensus is None:
        return False
    resolved = _resolved_side(metrics)
    return resolved is None or consensus == resolved
