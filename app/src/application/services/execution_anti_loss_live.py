"""Ramo live do gate anti-loss: flip/EMA/confirm e RSI soft pos-lado."""

from __future__ import annotations

from typing import Any

from src.application.services.execution_anti_loss_helpers import (
    check_mini_ema_trend_and_slope,
    check_rsi_filter,
    finalize_anti_loss_decision,
)
from src.application.services.execution_fusion_p_eff import sync_fusion_p_eff_for_direction
from src.application.services.execution_neg_edge import _min_edge_from_orch, _payout_from_orch
from src.application.services.market_audit_log_helpers import resolve_predicted_edge
from src.domain.models.trade import TradeDirection


_VALID = {TradeDirection.CALL.name, TradeDirection.PUT.name}


def _side(value: object) -> str | None:
    """Normaliza CALL/PUT ou None."""
    side = str(value or "").strip().upper()
    return side if side in _VALID else None


def _exec_dir(metrics: dict[str, Any]) -> TradeDirection | None:
    """Resolve direcao EXEC pos-fusao a partir de metrics."""
    name = _side(metrics.get("exec_direction"))
    return TradeDirection[name] if name is not None else None


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


def stamp_anti_loss_metrics(
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


def _rsi_soft_after_side(
    metrics: dict[str, Any],
    *,
    cfg: dict[str, Any],
    tcn: TradeDirection,
    candle: str | None,
    body: float | None,
    out: dict[str, Any],
) -> dict[str, Any]:
    """Aplica RSI soft Kelly no lado final; nao apaga flip/soft previo."""
    final = _exec_dir(metrics)
    if final is None:
        final = tcn
    rsi_min = float(cfg.get("anti_loss_rsi_min", 0.30))
    rsi_max = float(cfg.get("anti_loss_rsi_max", 0.70))
    if check_rsi_filter(metrics, final, rsi_min=rsi_min, rsi_max=rsi_max):
        return out
    metrics["anti_loss_rsi_soft"] = True
    if bool(out.get("soft")) or bool(out.get("active")):
        return out
    stamp_anti_loss_metrics(metrics, tcn=tcn, candle=candle, body=body, reason="anti_loss_rsi_momentum", side=final)
    return finalize_anti_loss_decision(out, cfg=cfg, reason="anti_loss_rsi_momentum")


def _candle_flip_edge_ok(
    metrics: dict[str, Any],
    candle: str,
    *,
    orch: Any | None,
) -> bool:
    """True se Edge Cal do lado da vela >= piso explore/recovery."""
    pay = _payout_from_orch(orch)
    min_edge = float(_min_edge_from_orch(orch, metrics=metrics))
    if min_edge <= 0.0:
        min_edge = 0.015
    edge_candle = float(resolve_predicted_edge(metrics, direction=candle, payout=pay))
    metrics["anti_loss_flip_candle_edge"] = edge_candle
    metrics["anti_loss_flip_min_edge"] = min_edge
    if edge_candle + 1e-12 < min_edge:
        metrics["anti_loss_flip_blocked"] = "edge_nonpos" if edge_candle + 1e-12 <= 0.0 else "edge_subfloor"
        return False
    metrics.pop("anti_loss_flip_blocked", None)
    return True


def _apply_candle_flip(
    metrics: dict[str, Any],
    *,
    cfg: dict[str, Any],
    candle: str,
    out: dict[str, Any],
) -> dict[str, Any]:
    """Inverte EXEC para o lado da vela e sincroniza fusion_p_eff."""
    new_side = TradeDirection[candle]
    metrics["exec_direction"], metrics["resolved_direction"] = new_side.name, new_side.name
    sync_fusion_p_eff_for_direction(metrics, new_side.name)
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


def evaluate_live_anti_loss(
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
    """Ramo live: flip so com Edge vela >= floor; hybrid discord soft; RSI soft pos-lado."""
    out = {"active": False, "skip": False, "soft": False, "reason": None, "soft_mult": None}
    exec_side = _exec_dir(metrics)
    anchor = exec_side if exec_side is not None else tcn
    allow_flip = bool(cfg.get("anti_loss_allow_candle_flip", True))
    last_dir = _side(metrics.get("closed_micro_candle_dir"))
    hybrid_disagree = (
        metrics.get("anti_loss_anchor_agree") is False and last_dir is not None and last_dir != anchor.name
    )
    if bool(cfg.get("anti_loss_live_exec_candle_enabled", False)) and candle is not None and anchor.name != candle:
        stamp_anti_loss_metrics(metrics, tcn=tcn, candle=candle, body=body, reason="live_exec_discord", side=anchor)
        if allow_flip and candle in _VALID and _candle_flip_edge_ok(metrics, candle, orch=orch):
            return _rsi_soft_after_side(
                metrics,
                cfg=cfg,
                tcn=tcn,
                candle=candle,
                body=body,
                out=_apply_candle_flip(metrics, cfg=cfg, candle=candle, out=out),
            )
        return finalize_anti_loss_decision(out, cfg=cfg, reason="live_exec_discord")
    if hybrid_disagree and last_dir is not None:
        stamp_anti_loss_metrics(metrics, tcn=tcn, candle=last_dir, body=body, reason="live_discord_weak", side=anchor)
        if allow_flip and _candle_flip_edge_ok(metrics, last_dir, orch=orch):
            return _rsi_soft_after_side(
                metrics,
                cfg=cfg,
                tcn=tcn,
                candle=last_dir,
                body=body,
                out=_apply_candle_flip(metrics, cfg=cfg, candle=last_dir, out=out),
            )
        return _rsi_soft_after_side(
            metrics,
            cfg=cfg,
            tcn=tcn,
            candle=last_dir,
            body=body,
            out=finalize_anti_loss_decision(out, cfg=cfg, reason="live_discord_weak"),
        )
    ema_ok, ema_reason = check_mini_ema_trend_and_slope(orch, symbol, anchor, metrics=metrics)
    if not ema_ok:
        why = ema_reason or "anti_loss_ema_trend"
        stamp_anti_loss_metrics(metrics, tcn=tcn, candle=candle, body=body, reason=why, side=anchor)
        if (
            allow_flip
            and candle is not None
            and candle in _VALID
            and anchor.name != candle
            and _candle_flip_edge_ok(metrics, candle, orch=orch)
        ):
            return _rsi_soft_after_side(
                metrics,
                cfg=cfg,
                tcn=tcn,
                candle=candle,
                body=body,
                out=_apply_candle_flip(metrics, cfg=cfg, candle=candle, out=out),
            )
        return _rsi_soft_after_side(
            metrics,
            cfg=cfg,
            tcn=tcn,
            candle=candle,
            body=body,
            out=finalize_anti_loss_decision(out, cfg=cfg, reason=why),
        )
    atr_val = metrics.get("atr")
    effective_min_body = (
        max(min_body, float(atr_val) * 0.5) if atr_val is not None and float(atr_val) > 0.0 else min_body
    )
    if bool(cfg.get("anti_loss_live_weak_candle_enabled", True)) and _weak_candle(candle, body, effective_min_body):
        reason = _live_weak_reason(candle)
        stamp_anti_loss_metrics(metrics, tcn=tcn, candle=candle, body=body, reason=reason, side=anchor)
        return _rsi_soft_after_side(
            metrics,
            cfg=cfg,
            tcn=tcn,
            candle=candle,
            body=body,
            out=finalize_anti_loss_decision(out, cfg=cfg, reason=reason),
        )
    confirm_min = float(cfg.get("anti_loss_live_confirm_min_body", 0.05))
    if bool(cfg.get("anti_loss_live_confirm_enabled", True)) and not _agree_strong(candle, anchor, body, confirm_min):
        reason = _live_confirm_reason(candle, anchor)
        stamp_anti_loss_metrics(metrics, tcn=tcn, candle=candle, body=body, reason=reason, side=anchor)
        return _rsi_soft_after_side(
            metrics,
            cfg=cfg,
            tcn=tcn,
            candle=candle,
            body=body,
            out=finalize_anti_loss_decision(out, cfg=cfg, reason=reason),
        )
    return _rsi_soft_after_side(metrics, cfg=cfg, tcn=tcn, candle=candle, body=body, out=out)
