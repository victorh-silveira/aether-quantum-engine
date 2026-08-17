"""Gate anti-loss: seed + p_loss alto + vela discordante do TCN."""

from __future__ import annotations

from math import isfinite
from typing import Any

from src.application.services.execution_signal_skip import apply_kelly_soft, parse_signal_skip_config
from src.application.services.loss_classifier_flip import closed_micro_candle_side, tcn_pos_edge_blocks_flip
from src.domain.models.trade import TradeDirection


__all__ = ("apply_anti_loss_seed_discord", "evaluate_anti_loss_seed_discord")

_VALID = {TradeDirection.CALL.name, TradeDirection.PUT.name}
_GATE = "anti_loss_seed_discord"


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


def _candle_body(metrics: dict[str, Any]) -> float | None:
    """Le closed_micro_candle_body numerico ou None."""
    raw = metrics.get("closed_micro_candle_body")
    if raw is None:
        return None
    try:
        body = float(raw)
    except (TypeError, ValueError):
        return None
    if not isfinite(body) or body < 0.0:
        return None
    return body


def _agree_strong(candle: str | None, tcn: TradeDirection, body: float | None, min_body: float) -> bool:
    """True se vela == TCN e |c-o| atinge o piso de corpo."""
    if candle is None or candle != tcn.name or body is None:
        return False
    return body + 1e-12 >= float(min_body)


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
    }
    return bool(tcn_pos_edge_blocks_flip(metrics, tcn, cfg=block_cfg))


def evaluate_anti_loss_seed_discord(
    metrics: dict[str, Any],
    *,
    cfg: dict[str, Any],
    orch: Any | None = None,
) -> dict[str, Any]:
    """Decide SKIP duro (EXPLORE e RECOVER) se a vela nao confirma o TCN com corpo minimo."""
    _ = orch
    out = {"active": False, "skip": False, "soft": False, "reason": None, "soft_mult": None}
    if not bool(cfg.get("anti_loss_seed_discord_enabled", False)):
        return out
    if bool(cfg.get("anti_loss_require_seed", True)) and bool(metrics.get("loss_clf_auto_learn")):
        return out
    p_loss = _p_loss(metrics)
    if p_loss is None:
        return out
    floor = float(cfg.get("anti_loss_p_loss_floor", 0.85))
    if p_loss + 1e-12 < floor:
        return out
    tcn = _tcn_dir(metrics)
    if tcn is None:
        return out
    if bool(cfg.get("anti_loss_require_tcn_pos_edge", True)) and not _tcn_pos_edge_locked(metrics, tcn):
        return out
    candle = closed_micro_candle_side(metrics)
    body = _candle_body(metrics)
    min_body = float(cfg.get("anti_loss_min_candle_body", 0.10))
    if _agree_strong(candle, tcn, body, min_body):
        return out
    out["active"] = True
    out["reason"] = "seed_weak_candle" if candle == tcn.name else "seed_discord"
    metrics["anti_loss_p_loss"] = p_loss
    metrics["anti_loss_tcn"] = tcn.name
    metrics["anti_loss_candle"] = candle or "-"
    if body is not None:
        metrics["anti_loss_body"] = body
    if bool(cfg.get("anti_loss_hard_skip", True)):
        out["skip"] = True
        return out
    out["soft"] = True
    out["soft_mult"] = float(cfg.get("anti_loss_soft_kelly_mult", 0.25))
    return out


def apply_anti_loss_seed_discord(
    metrics: dict[str, Any],
    *,
    orch: Any | None = None,
    force: bool = False,
    cfg: dict[str, Any] | None = None,
) -> bool:
    """Aplica SKIP duro seed+discord (tambem com PEND); True se skipou EXEC."""
    if force:
        return False
    if metrics.get("execution_candidate_ready") is False:
        return False
    vision = cfg if isinstance(cfg, dict) else parse_signal_skip_config(None)
    decision = evaluate_anti_loss_seed_discord(metrics, cfg=vision, orch=orch)
    if not decision["active"]:
        metrics.pop("anti_loss_seed_discord", None)
        metrics.pop("anti_loss_soft", None)
        return False
    metrics["anti_loss_seed_discord"] = True
    metrics["anti_loss_why"] = str(decision.get("reason") or "seed_discord")
    if decision["skip"]:
        metrics["execution_candidate_ready"] = False
        metrics["gate_reason"] = _GATE
        metrics["signal_status"] = "SKIP:ANTI_LOSS_SEED_DISCORD"
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
