"""Helpers de adapt SCALE por retracao, explosao e mili+tape (sem SKIP)."""

from __future__ import annotations

from typing import Any

from src.application.services.execution_scale_micro import classify_micro_regime
from src.domain.models.trade import TradeDirection


_VALID = {TradeDirection.CALL.name, TradeDirection.PUT.name}


def _side(value: object) -> str | None:
    """Normaliza CALL/PUT ou None."""
    side = str(value or "").strip().upper()
    return side if side in _VALID else None


def _flip_to(
    metrics: dict[str, Any],
    exec_dir: TradeDirection,
    live: str,
    *,
    reason: str,
) -> TradeDirection | None:
    """Marca adaptacao quando o lado vivo discrepa do TCN."""
    if live not in _VALID or live == exec_dir.name:
        return None
    metrics["scale_adapted"] = True
    metrics["scale_adapt_reason"] = reason
    return TradeDirection[live]


def adapt_on_retraction(
    metrics: dict[str, Any], exec_dir: TradeDirection, cfg: dict[str, Any]
) -> TradeDirection | None:
    """Adapta ao lado vivo quando ha retracao confirmada contra o TCN."""
    if not bool(cfg.get("adapt_on_retraction", True)):
        return None
    classify_micro_regime(metrics, exec_dir.name, cfg=cfg)
    if not bool(metrics.get("scale_retraction_vs_tcn")):
        return None
    live = _side(metrics.get("scale_micro_side"))
    if live is None:
        return None
    return _flip_to(metrics, exec_dir, live, reason="retraction")


def adapt_on_explosion(metrics: dict[str, Any], exec_dir: TradeDirection, cfg: dict[str, Any]) -> TradeDirection | None:
    """Adapta quando explosao MINI+MILI aponta lado oposto ao TCN."""
    if not bool(cfg.get("adapt_on_explosion", True)):
        return None
    classify_micro_regime(metrics, exec_dir.name, cfg=cfg)
    if str(metrics.get("scale_micro_regime") or "").lower() != "explosion":
        return None
    live = _side(metrics.get("scale_micro_side"))
    if live is None:
        return None
    return _flip_to(metrics, exec_dir, live, reason="explosion_vs_tcn")


def adapt_on_mili_tape(metrics: dict[str, Any], exec_dir: TradeDirection, cfg: dict[str, Any]) -> TradeDirection | None:
    """Adapta quando MILI e tape concordam contra o TCN (fora de chop se knob ativo)."""
    if not bool(cfg.get("adapt_on_mili_tape", True)):
        return None
    classify_micro_regime(metrics, exec_dir.name, cfg=cfg)
    if (
        bool(cfg.get("adapt_mili_tape_skip_chop", True))
        and str(metrics.get("scale_micro_regime") or "").lower() == "chop"
    ):
        return None
    mili = _side(metrics.get("scale_mili_dir"))
    tape = _side(metrics.get("scale_tape_consensus"))
    if mili is None or tape is None or mili != tape:
        return None
    return _flip_to(metrics, exec_dir, mili, reason="mili_tape_vs_tcn")


def try_regime_adapts(metrics: dict[str, Any], exec_dir: TradeDirection, cfg: dict[str, Any]) -> TradeDirection | None:
    """Tenta retracao, explosao e mili+tape nesta ordem."""
    for adapter in (adapt_on_retraction, adapt_on_explosion, adapt_on_mili_tape):
        adapted = adapter(metrics, exec_dir, cfg)
        if adapted is not None:
            return adapted
    return None
