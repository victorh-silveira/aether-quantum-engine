"""Gate anti-loss: seed + p_loss alto + vela fraca live."""

from __future__ import annotations

from typing import Any

from src.application.services.execution_anti_loss_helpers import finalize_anti_loss_decision
from src.application.services.execution_anti_loss_live import evaluate_live_anti_loss, stamp_anti_loss_metrics
from src.application.services.execution_gate_verdict import stamp_hard_skip
from src.application.services.execution_signal_skip import apply_kelly_soft, parse_signal_skip_config
from src.application.services.loss_classifier_flip import tcn_pos_edge_blocks_flip
from src.application.services.market_audit_ops_window import (
    ops_window_candle_body,
    ops_window_candle_side,
    ops_window_stamped,
    resolve_hybrid_candle_anchor,
)
from src.domain.models.trade import TradeDirection


__all__ = ("apply_anti_loss_seed_discord", "evaluate_anti_loss_seed_discord")

_VALID = {TradeDirection.CALL.name, TradeDirection.PUT.name}
_GATE = "anti_loss_seed_discord"
_KNOWN_REASONS = {
    "anti_loss_rsi_momentum",
    "anti_loss_rsi_trend",
    "anti_loss_ema_trend",
    "anti_loss_ema_slope",
    "live_exec_discord",
    "live_discord_weak",
    "live_confirm_weak",
    "live_weak_candle",
    "live_no_candle",
}


def _side(value: object) -> str | None:
    """Normaliza CALL/PUT ou None."""
    side = str(value or "").strip().upper()
    return side if side in _VALID else None


def _p_loss(metrics: dict[str, Any]) -> float | None:
    """Le loss_clf_p_loss numerico ou None."""
    raw = metrics.get("loss_clf_p_loss")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _tcn_dir(metrics: dict[str, Any]) -> TradeDirection | None:
    """Resolve direcao TCN a partir de metrics."""
    name = _side(metrics.get("tcn_direction") or metrics.get("resolved_direction"))
    return TradeDirection[name] if name is not None else None


def _agree_strong(candle: str | None, side: TradeDirection, body: float | None, min_body: float) -> bool:
    """True se vela == lado ancora e corpo atinge o piso."""
    return candle == side.name and body is not None and body + 1e-12 >= float(min_body)


def _tcn_pos_edge_locked(metrics: dict[str, Any], tcn: TradeDirection) -> bool:
    """True se fusao/loss-clf ja travaram TCN por pos_edge Cal+raw."""
    if bool(metrics.get("fusion_blocked_tcn_pos_edge")) or bool(metrics.get("loss_clf_flip_block_tcn_pos_edge")):
        return True
    if str(metrics.get("fusion_reason") or "").strip() == "tcn_pos_edge":
        return True
    block_cfg = {
        "flip_block_when_tcn_pos_edge": True,
        "flip_min_edge_execute": 0.04,
        "flip_tcn_pos_edge_raw_floor": 0.04,
        "flip_waive_tcn_pos_edge_on_discord": False,
    }
    return bool(tcn_pos_edge_blocks_flip(metrics, tcn, cfg=block_cfg))


def evaluate_anti_loss_seed_discord(
    metrics: dict[str, Any],
    *,
    cfg: dict[str, Any],
    orch: Any | None = None,
    symbol: str | None = None,
) -> dict[str, Any]:
    """Decide SKIP se vela live fraca ou seed+discord sem confirmacao forte."""
    out = {"active": False, "skip": False, "soft": False, "reason": None, "soft_mult": None}
    if not bool(cfg.get("anti_loss_seed_discord_enabled", True)):
        return out
    tcn = _tcn_dir(metrics)
    if tcn is None:
        return out
    hybrid_side, hybrid_body, hybrid_agree = resolve_hybrid_candle_anchor(metrics)
    candle = ops_window_candle_side(metrics)
    body = ops_window_candle_body(metrics)
    min_body = float(cfg.get("anti_loss_min_candle_body", 0.10))
    metrics["anti_loss_ops_dir"] = candle
    metrics["anti_loss_last_dir"] = str(metrics.get("closed_micro_candle_dir") or "-")
    if ops_window_stamped(metrics):
        metrics["anti_loss_anchor_mode"] = "hybrid"
        metrics["anti_loss_anchor_agree"] = hybrid_agree
        return evaluate_live_anti_loss(
            metrics,
            cfg=cfg,
            tcn=tcn,
            candle=hybrid_side if hybrid_side is not None else candle,
            body=hybrid_body if hybrid_body is not None else body,
            min_body=min_body,
            orch=orch,
            symbol=symbol,
        )
    metrics["anti_loss_anchor_mode"] = "ops_window"
    metrics["anti_loss_anchor_agree"] = False
    if bool(metrics.get("loss_clf_auto_learn")):
        return out
    p_loss = _p_loss(metrics)
    if p_loss is None:
        return out
    floor = float(cfg.get("anti_loss_p_loss_floor", 0.85))
    if p_loss + 1e-12 < floor:
        return out
    if bool(cfg.get("anti_loss_require_tcn_pos_edge", True)) and not _tcn_pos_edge_locked(metrics, tcn):
        return out
    if _agree_strong(candle, tcn, body, min_body):
        return out
    reason = "seed_weak_candle" if candle == tcn.name else "seed_discord"
    stamp_anti_loss_metrics(metrics, tcn=tcn, candle=candle, body=body, reason=reason, p_loss=p_loss)
    return finalize_anti_loss_decision(out, cfg=cfg, reason=reason)


def apply_anti_loss_seed_discord(
    metrics: dict[str, Any],
    *,
    orch: Any | None = None,
    force: bool = False,
    cfg: dict[str, Any] | None = None,
    symbol: str | None = None,
) -> bool:
    """Aplica SKIP duro seed+discord (tambem com PEND); True se skipou EXEC."""
    if force:
        return False
    if metrics.get("execution_candidate_ready") is False:
        return False
    vision = cfg if isinstance(cfg, dict) else parse_signal_skip_config(None)
    sym = (
        symbol or getattr(orch, "anchor", None) or list(getattr(orch, "symbols", []))[0]
        if getattr(orch, "symbols", None)
        else None
    )
    decision = evaluate_anti_loss_seed_discord(metrics, cfg=vision, orch=orch, symbol=sym)
    if not decision["active"]:
        metrics.pop("anti_loss_seed_discord", None)
        metrics.pop("anti_loss_soft", None)
        return False
    metrics["anti_loss_seed_discord"] = True
    reason = str(decision.get("reason") or "seed_discord")
    metrics["anti_loss_why"] = reason
    if decision["skip"]:
        metrics["execution_candidate_ready"] = False
        metrics["gate_reason"] = reason if reason in _KNOWN_REASONS else _GATE
        metrics["signal_status"] = f"SKIP:{metrics['gate_reason'].upper()}"
        metrics.pop("anti_loss_soft", None)
        stamp_hard_skip(metrics, str(metrics["gate_reason"]))
        return True
    if decision["soft"]:
        apply_kelly_soft(
            metrics,
            float(decision["soft_mult"] or 0.25),
            waived="anti_loss_soft",
            flag="anti_loss_soft",
        )
    return False
