"""Gates booleanos de microestrutura: discord vela/tape e chop+p_loss — HARD sem flip."""

from __future__ import annotations

import logging
from typing import Any

from src.application.services.execution_gate_verdict import stamp_hard_skip
from src.domain.config_knobs import merge_settings_block, require_bool, require_float, require_keys
from src.domain.models.trade import TradeDirection


logger = logging.getLogger("AETH")

_VALID = {TradeDirection.CALL.name, TradeDirection.PUT.name}


def parse_micro_protect_config(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve knobs micro_discord / chop_loss_risk em signal_skip."""
    block = merge_settings_block(("orchestrator", "execution", "signal_skip"), raw)
    require_keys(
        block,
        (
            "micro_discord_hard_skip",
            "micro_discord_min_body",
            "chop_loss_risk_hard_skip",
            "chop_loss_risk_p_loss_floor",
        ),
        "orchestrator.execution.signal_skip",
    )
    min_body = require_float(block, "micro_discord_min_body")
    if min_body < 0.0:
        raise ValueError("orchestrator.execution.signal_skip.micro_discord_min_body deve ser >= 0")
    p_floor = require_float(block, "chop_loss_risk_p_loss_floor")
    if p_floor < 0.0 or p_floor > 1.0:
        raise ValueError("orchestrator.execution.signal_skip.chop_loss_risk_p_loss_floor deve estar em [0, 1]")
    return {
        "micro_discord_hard_skip": require_bool(block, "micro_discord_hard_skip"),
        "micro_discord_min_body": min_body,
        "chop_loss_risk_hard_skip": require_bool(block, "chop_loss_risk_hard_skip"),
        "chop_loss_risk_p_loss_floor": p_floor,
    }


def _side(value: object) -> str | None:
    """Normaliza CALL/PUT."""
    name = str(value or "").strip().upper()
    return name if name in _VALID else None


def _body(metrics: dict[str, Any], key: str) -> float | None:
    """Le corpo numerico; None se ausente."""
    raw = metrics.get(key)
    if raw is None:
        return None
    try:
        body = float(raw)
    except (TypeError, ValueError):
        return None
    return body if body > 0.0 else None


def _confirm_candle_discord(metrics: dict[str, Any], candle: str, exec_side: str) -> bool:
    """True se tape/ops/mi confirmam a vela contra o EXEC (sem inverter lado)."""
    votes = 0
    for key in (
        "scale_tape_consensus",
        "ops_window_candle_dir",
        "scale_mini_prev_bar_dir",
        "scale_mini_bar_dir",
        "scale_mini_dir",
        "scale_mili_dir",
    ):
        side = _side(metrics.get(key))
        if side is None:
            continue
        if side == candle:
            votes += 1
        elif side == exec_side:
            votes -= 1
    return votes > 0


def _stamp(metrics: dict[str, Any], reason: str, *, orch: Any | None) -> None:
    """Aplica HARD SKIP preservando exec_direction."""
    metrics["execution_candidate_ready"] = False
    metrics["gate_reason"] = reason
    metrics["signal_status"] = f"SKIP:{reason.upper()}"
    stamp_hard_skip(metrics, reason)
    if orch is not None:
        logger.info(
            "MICRO || HARD_SKIP why=%s exec=%s candle=%s regime=%s p_loss=%s",
            reason,
            str(metrics.get("exec_direction") or "-"),
            str(metrics.get("closed_micro_candle_dir") or "-"),
            str(metrics.get("scale_micro_regime") or "-"),
            metrics.get("loss_clf_p_loss"),
        )


def apply_micro_discord_hard_skip(
    metrics: dict[str, Any],
    *,
    orch: Any | None = None,
    force: bool = False,
    cfg: dict[str, Any] | None = None,
) -> bool:
    """HARD se vela M5 fechada discorda do EXEC e ha confirmacao tape/ops — sem flip."""
    if force:
        return False
    if metrics.get("execution_candidate_ready") is False:
        return False
    status = str(metrics.get("signal_status") or "").strip().upper()
    if status == "SKIP" or status.startswith("SKIP:"):
        return False
    vision = cfg if isinstance(cfg, dict) and "micro_discord_hard_skip" in cfg else parse_micro_protect_config(cfg)
    if not bool(vision.get("micro_discord_hard_skip", False)):
        return False
    exec_side = _side(metrics.get("exec_direction") or metrics.get("resolved_direction"))
    candle = _side(metrics.get("closed_micro_candle_dir") or metrics.get("scale_micro_bar_dir"))
    if exec_side is None or candle is None or candle == exec_side:
        return False
    body = _body(metrics, "closed_micro_candle_body")
    if body is None:
        body = _body(metrics, "ops_window_candle_body")
    min_body = float(vision["micro_discord_min_body"])
    if body is None or body + 1e-12 < min_body:
        return False
    if not _confirm_candle_discord(metrics, candle, exec_side):
        return False
    metrics["micro_discord_candle"] = candle
    metrics["micro_discord_exec"] = exec_side
    metrics["micro_discord_body"] = float(body)
    _stamp(metrics, "micro_discord", orch=orch)
    return True


def apply_chop_loss_risk_hard_skip(
    metrics: dict[str, Any],
    *,
    orch: Any | None = None,
    force: bool = False,
    cfg: dict[str, Any] | None = None,
) -> bool:
    """HARD se micro=chop e loss-clf p_loss alto (seed soft/FLIP_BLOCK) — sem flip."""
    if force:
        return False
    if metrics.get("execution_candidate_ready") is False:
        return False
    status = str(metrics.get("signal_status") or "").strip().upper()
    if status == "SKIP" or status.startswith("SKIP:"):
        return False
    vision = cfg if isinstance(cfg, dict) and "chop_loss_risk_hard_skip" in cfg else parse_micro_protect_config(cfg)
    if not bool(vision.get("chop_loss_risk_hard_skip", False)):
        return False
    regime = str(metrics.get("scale_micro_regime") or "").strip().lower()
    if regime != "chop":
        return False
    try:
        p_loss = float(metrics.get("loss_clf_p_loss"))
    except (TypeError, ValueError):
        return False
    floor = float(vision["chop_loss_risk_p_loss_floor"])
    if p_loss + 1e-12 < floor:
        return False
    soft = bool(metrics.get("loss_clf_soft"))
    blocked = str(metrics.get("loss_clf_flip_blocked") or "").strip()
    if not soft and not blocked:
        return False
    metrics["chop_loss_risk_p_loss"] = p_loss
    _stamp(metrics, "chop_loss_risk", orch=orch)
    return True


def apply_micro_protect_gates(
    metrics: dict[str, Any],
    *,
    orch: Any | None = None,
    force: bool = False,
    cfg: dict[str, Any] | None = None,
) -> bool:
    """Aplica discord e chop+p_loss; retorna True se HARD SKIP."""
    vision = cfg if isinstance(cfg, dict) and "micro_discord_hard_skip" in cfg else parse_micro_protect_config(cfg)
    if apply_micro_discord_hard_skip(metrics, orch=orch, force=force, cfg=vision):
        return True
    return apply_chop_loss_risk_hard_skip(metrics, orch=orch, force=force, cfg=vision)
