"""Gates booleanos de microestrutura: discord vela, soft+p_loss e score Soft."""

from __future__ import annotations

import logging
from typing import Any

from src.application.services.execution_gate_verdict import stamp_hard_skip
from src.application.services.execution_micro_follow import apply_micro_discord_follow_candle
from src.domain.config_knobs import merge_settings_block, require_bool, require_float, require_int, require_keys
from src.domain.models.trade import TradeDirection


logger = logging.getLogger("AETH")

_VALID = {TradeDirection.CALL.name, TradeDirection.PUT.name}


def parse_micro_protect_config(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve knobs micro_discord / chop_loss_risk / soft_confirm em signal_skip."""
    block = merge_settings_block(("orchestrator", "execution", "signal_skip"), raw)
    require_keys(
        block,
        (
            "micro_discord_hard_skip",
            "micro_discord_min_body",
            "micro_discord_follow_candle",
            "micro_discord_follow_kelly_mult",
            "chop_loss_risk_hard_skip",
            "chop_loss_risk_p_loss_floor",
            "soft_confirm_weak_hard_skip",
            "soft_exec_min_confirmations",
        ),
        "orchestrator.execution.signal_skip",
    )
    min_body = require_float(block, "micro_discord_min_body")
    if min_body < 0.0:
        raise ValueError("orchestrator.execution.signal_skip.micro_discord_min_body deve ser >= 0")
    p_floor = require_float(block, "chop_loss_risk_p_loss_floor")
    if p_floor < 0.0 or p_floor > 1.0:
        raise ValueError("orchestrator.execution.signal_skip.chop_loss_risk_p_loss_floor deve estar em [0, 1]")
    follow_mult = require_float(block, "micro_discord_follow_kelly_mult")
    if follow_mult <= 0.0 or follow_mult > 1.0:
        raise ValueError("orchestrator.execution.signal_skip.micro_discord_follow_kelly_mult deve estar em (0, 1]")
    min_conf = require_int(block, "soft_exec_min_confirmations")
    if min_conf < 1:
        raise ValueError("orchestrator.execution.signal_skip.soft_exec_min_confirmations deve ser >= 1")
    return {
        "micro_discord_hard_skip": require_bool(block, "micro_discord_hard_skip"),
        "micro_discord_min_body": min_body,
        "micro_discord_follow_candle": require_bool(block, "micro_discord_follow_candle"),
        "micro_discord_follow_kelly_mult": follow_mult,
        "chop_loss_risk_hard_skip": require_bool(block, "chop_loss_risk_hard_skip"),
        "chop_loss_risk_p_loss_floor": p_floor,
        "soft_confirm_weak_hard_skip": require_bool(block, "soft_confirm_weak_hard_skip"),
        "soft_exec_min_confirmations": min_conf,
        "min_edge_explore": float(block.get("min_edge_explore") or 0.015),
        "anti_loss_soft_kelly_mult": float(block.get("anti_loss_soft_kelly_mult") or follow_mult),
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


def _soft_or_flip_blocked(metrics: dict[str, Any]) -> bool:
    """True se loss-clf soft ou FLIP_BLOCK ativo."""
    if bool(metrics.get("loss_clf_soft")):
        return True
    return bool(str(metrics.get("loss_clf_flip_blocked") or "").strip())


def score_soft_confirmations(metrics: dict[str, Any], exec_side: str) -> tuple[int, list[str]]:
    """Conta peers definidos que concordam com EXEC; retorna (score, nomes que bateram)."""
    peers: list[tuple[str, str | None]] = [
        ("candle", _side(metrics.get("closed_micro_candle_dir") or metrics.get("scale_micro_bar_dir"))),
        ("tape", _side(metrics.get("scale_tape_consensus"))),
        ("mini", _side(metrics.get("scale_mini_bar_dir")) or _side(metrics.get("scale_mini_dir"))),
        ("mili", _side(metrics.get("scale_mili_dir"))),
        ("ops", _side(metrics.get("ops_window_candle_dir"))),
    ]
    hit: list[str] = []
    for name, side in peers:
        if side is None:
            continue
        if side == exec_side:
            hit.append(name)
    return len(hit), hit


def _stamp(metrics: dict[str, Any], reason: str, *, orch: Any | None) -> None:
    """Aplica HARD SKIP preservando exec_direction."""
    metrics["execution_candidate_ready"] = False
    metrics["gate_reason"] = reason
    metrics["signal_status"] = f"SKIP:{reason.upper()}"
    stamp_hard_skip(metrics, reason)
    if orch is not None:
        logger.debug(
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
    """Discord vela×EXEC: follow Soft_SIZE se Edge vela>=piso; senao HARD sem flip."""
    if force:
        return False
    if bool(metrics.get("loss_clf_flip")):
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
    metrics["micro_discord_confirmed"] = _confirm_candle_discord(metrics, candle, exec_side)
    metrics["micro_discord_candle"] = candle
    metrics["micro_discord_exec"] = exec_side
    metrics["micro_discord_body"] = float(body)
    if apply_micro_discord_follow_candle(
        metrics,
        candle=candle,
        exec_side=exec_side,
        body=float(body),
        cfg=vision,
    ):
        if orch is not None:
            logger.info(
                "MICRO || FOLLOW why=micro_discord_follow from=%s to=%s body=%.3f edge=%s",
                exec_side,
                candle,
                float(body),
                metrics.get("micro_discord_follow_candle_edge"),
            )
        return False
    _stamp(metrics, "micro_discord", orch=orch)
    return True


def apply_chop_loss_risk_hard_skip(
    metrics: dict[str, Any],
    *,
    orch: Any | None = None,
    force: bool = False,
    cfg: dict[str, Any] | None = None,
) -> bool:
    """HARD se soft/FLIP_BLOCK, p_loss alto e vela M5 discorda do EXEC — sem flip."""
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
    try:
        p_loss = float(metrics.get("loss_clf_p_loss"))
    except (TypeError, ValueError):
        return False
    floor = float(vision["chop_loss_risk_p_loss_floor"])
    if p_loss + 1e-12 < floor:
        return False
    if not _soft_or_flip_blocked(metrics):
        return False
    exec_side = _side(metrics.get("exec_direction") or metrics.get("resolved_direction"))
    candle = _side(metrics.get("closed_micro_candle_dir") or metrics.get("scale_micro_bar_dir"))
    if exec_side is None or candle is None or candle == exec_side:
        return False
    metrics["chop_loss_risk_p_loss"] = p_loss
    metrics["chop_loss_risk_candle"] = candle
    metrics["chop_loss_risk_exec"] = exec_side
    _stamp(metrics, "chop_loss_risk", orch=orch)
    return True


def apply_soft_confirm_weak_hard_skip(
    metrics: dict[str, Any],
    *,
    orch: Any | None = None,
    force: bool = False,
    cfg: dict[str, Any] | None = None,
) -> bool:
    """HARD se soft/FLIP_BLOCK e confirm_score < min — sem flip."""
    if force:
        return False
    if metrics.get("execution_candidate_ready") is False:
        return False
    status = str(metrics.get("signal_status") or "").strip().upper()
    if status == "SKIP" or status.startswith("SKIP:"):
        return False
    vision = cfg if isinstance(cfg, dict) and "soft_confirm_weak_hard_skip" in cfg else parse_micro_protect_config(cfg)
    if not bool(vision.get("soft_confirm_weak_hard_skip", False)):
        return False
    if not _soft_or_flip_blocked(metrics):
        return False
    exec_side = _side(metrics.get("exec_direction") or metrics.get("resolved_direction"))
    if exec_side is None:
        return False
    score, peers = score_soft_confirmations(metrics, exec_side)
    metrics["soft_confirm_score"] = score
    metrics["soft_confirm_peers"] = ",".join(peers) if peers else "-"
    min_conf = int(vision["soft_exec_min_confirmations"])
    if score >= min_conf:
        return False
    _stamp(metrics, "soft_confirm_weak", orch=orch)
    return True


def apply_micro_protect_gates(
    metrics: dict[str, Any],
    *,
    orch: Any | None = None,
    force: bool = False,
    cfg: dict[str, Any] | None = None,
) -> bool:
    """Aplica discord, soft+p_loss e soft_confirm_weak; retorna True se HARD SKIP."""
    vision = cfg if isinstance(cfg, dict) and "micro_discord_hard_skip" in cfg else parse_micro_protect_config(cfg)
    if apply_micro_discord_hard_skip(metrics, orch=orch, force=force, cfg=vision):
        return True
    if apply_chop_loss_risk_hard_skip(metrics, orch=orch, force=force, cfg=vision):
        return True
    return apply_soft_confirm_weak_hard_skip(metrics, orch=orch, force=force, cfg=vision)
