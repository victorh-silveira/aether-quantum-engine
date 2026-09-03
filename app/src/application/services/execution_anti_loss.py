"""Gate anti-loss: seed + p_loss alto + vela fraca live."""

from __future__ import annotations

from typing import Any

from src.application.services.execution_anti_loss_helpers import (
    check_mini_ema_trend_and_slope,
    check_rsi_filter,
)
from src.application.services.execution_signal_skip import apply_kelly_soft, parse_signal_skip_config
from src.application.services.loss_classifier_flip import tcn_pos_edge_blocks_flip
from src.application.services.market_audit_ops_window import (
    ops_window_candle_body,
    ops_window_candle_side,
    ops_window_stamped,
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
    if name is None:
        return None
    return TradeDirection[name]


def _exec_dir(metrics: dict[str, Any]) -> TradeDirection | None:
    """Resolve direcao EXEC pos-fusao a partir de metrics."""
    name = _side(metrics.get("exec_direction"))
    if name is None:
        return None
    return TradeDirection[name]


def _weak_candle(candle: str | None, body: float | None, min_body: float) -> bool:
    """True se vela ausente ou corpo abaixo do piso minimo."""
    return candle is None or body is None or body + 1e-12 < float(min_body)


def _agree_strong(candle: str | None, side: TradeDirection, body: float | None, min_body: float) -> bool:
    """True se vela == lado ancora e corpo atinge o piso."""
    return candle == side.name and body is not None and body + 1e-12 >= float(min_body)


def _live_weak_reason(candle: str | None) -> str:
    """Motivo telemetria para ramo live em vela fraca."""
    return "live_no_candle" if candle is None else "live_weak_candle"


def _live_confirm_reason(candle: str | None, side: TradeDirection) -> str:
    """Motivo telemetria quando a vela nao confirma o lado ancora no piso live."""
    return "live_discord_weak" if candle != side.name else "live_confirm_weak"


def _tcn_pos_edge_locked(metrics: dict[str, Any], tcn: TradeDirection) -> bool:
    """True se fusao/loss-clf ja travaram TCN por pos_edge Cal+raw."""
    if bool(metrics.get("fusion_blocked_tcn_pos_edge")):
        return True
    if bool(metrics.get("loss_clf_flip_block_tcn_pos_edge")):
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


def _stamp_anti_loss_metrics(
    metrics: dict[str, Any],
    *,
    tcn: TradeDirection,
    candle: str | None,
    body: float | None,
    reason: str,
    p_loss: float | None = None,
    side: TradeDirection | None = None,
) -> None:
    """Grava telemetria compartilhada do anti-loss."""
    if p_loss is not None:
        metrics["anti_loss_p_loss"] = p_loss
    metrics["anti_loss_tcn"] = tcn.name
    if side is not None:
        metrics["anti_loss_side"] = side.name
    metrics["anti_loss_candle"] = candle or "-"
    if body is not None:
        metrics["anti_loss_body"] = body
    metrics["anti_loss_why"] = reason


def _finalize_anti_loss_decision(out: dict[str, Any], *, cfg: dict[str, Any], reason: str) -> dict[str, Any]:
    """Marca decisao ativa e hard/soft conforme SSOT."""
    out["active"], out["reason"] = True, reason
    if bool(cfg.get("anti_loss_hard_skip", True)):
        out["skip"] = True
        return out
    out["soft"], out["soft_mult"] = True, float(cfg.get("anti_loss_soft_kelly_mult", 0.25))
    return out


def _evaluate_live_anti_loss(
    metrics: dict[str, Any],
    *,
    cfg: dict[str, Any],
    tcn: TradeDirection,
    candle: str | None,
    body: float | None,
    min_body: float,
    orch: Any | None = None,
    symbol: str | None = None,
) -> dict[str, Any]:
    """Ramo live: hard SKIP se vela contraria, EMA slope/trend contra, RSI momentum ou vela fraca."""
    out = {"active": False, "skip": False, "soft": False, "reason": None, "soft_mult": None}
    exec_side = _exec_dir(metrics)
    anchor = exec_side if exec_side is not None else tcn
    rsi_min = float(cfg.get("anti_loss_rsi_min", 0.35))
    rsi_max = float(cfg.get("anti_loss_rsi_max", 0.65))
    if not check_rsi_filter(metrics, anchor, rsi_min=rsi_min, rsi_max=rsi_max):
        _stamp_anti_loss_metrics(
            metrics, tcn=tcn, candle=candle, body=body, reason="anti_loss_rsi_momentum", side=anchor
        )
        return _finalize_anti_loss_decision(out, cfg=cfg, reason="anti_loss_rsi_momentum")
    allow_flip = bool(cfg.get("anti_loss_allow_candle_flip", False))
    if bool(cfg.get("anti_loss_live_exec_candle_enabled", False)) and candle is not None and anchor.name != candle:
        _stamp_anti_loss_metrics(metrics, tcn=tcn, candle=candle, body=body, reason="live_exec_discord", side=anchor)
        if allow_flip and candle in _VALID:
            new_side = TradeDirection[candle]
            metrics["exec_direction"], metrics["resolved_direction"] = new_side.name, new_side.name
            metrics["anti_loss_flipped_to_candle"], metrics["anti_loss_why"] = True, "live_exec_flip_to_candle"
            out.update(
                {
                    "active": True,
                    "reason": "live_exec_flip_to_candle",
                    "soft": True,
                    "soft_mult": float(cfg.get("anti_loss_soft_kelly_mult", 0.75)),
                }
            )
            return out
        return _finalize_anti_loss_decision(out, cfg=cfg, reason="live_exec_discord")
    ema_ok, ema_reason = check_mini_ema_trend_and_slope(orch, symbol, anchor, metrics=metrics)
    if not ema_ok:
        why = ema_reason or "anti_loss_ema_trend"
        _stamp_anti_loss_metrics(metrics, tcn=tcn, candle=candle, body=body, reason=why, side=anchor)
        if allow_flip and candle is not None and candle in _VALID and anchor.name != candle:
            new_side = TradeDirection[candle]
            metrics["exec_direction"], metrics["resolved_direction"] = new_side.name, new_side.name
            metrics["anti_loss_flipped_to_candle"], metrics["anti_loss_why"] = True, "live_exec_flip_to_candle"
            out.update(
                {
                    "active": True,
                    "reason": "live_exec_flip_to_candle",
                    "soft": True,
                    "soft_mult": float(cfg.get("anti_loss_soft_kelly_mult", 0.75)),
                }
            )
            return out
        return _finalize_anti_loss_decision(out, cfg=cfg, reason=why)
    atr_val = metrics.get("atr")
    effective_min_body = (
        max(min_body, float(atr_val) * 0.5) if atr_val is not None and float(atr_val) > 0.0 else min_body
    )
    if bool(cfg.get("anti_loss_live_weak_candle_enabled", True)) and _weak_candle(candle, body, effective_min_body):
        reason = _live_weak_reason(candle)
        _stamp_anti_loss_metrics(metrics, tcn=tcn, candle=candle, body=body, reason=reason, side=anchor)
        return _finalize_anti_loss_decision(out, cfg=cfg, reason=reason)
    confirm_min = float(cfg.get("anti_loss_live_confirm_min_body", 0.05))
    if bool(cfg.get("anti_loss_live_confirm_enabled", True)) and not _agree_strong(candle, anchor, body, confirm_min):
        reason = _live_confirm_reason(candle, anchor)
        _stamp_anti_loss_metrics(metrics, tcn=tcn, candle=candle, body=body, reason=reason, side=anchor)
        return _finalize_anti_loss_decision(out, cfg=cfg, reason=reason)
    return out


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
    candle = ops_window_candle_side(metrics)
    body = ops_window_candle_body(metrics)
    min_body = float(cfg.get("anti_loss_min_candle_body", 0.10))
    if ops_window_stamped(metrics):
        return _evaluate_live_anti_loss(
            metrics,
            cfg=cfg,
            tcn=tcn,
            candle=candle,
            body=body,
            min_body=min_body,
            orch=orch,
            symbol=symbol,
        )
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
    _stamp_anti_loss_metrics(metrics, tcn=tcn, candle=candle, body=body, reason=reason, p_loss=p_loss)
    return _finalize_anti_loss_decision(out, cfg=cfg, reason=reason)


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
        return True
    if decision["soft"]:
        apply_kelly_soft(
            metrics,
            float(decision["soft_mult"] or 0.25),
            waived="anti_loss_soft",
            flag="anti_loss_soft",
        )
    return False
