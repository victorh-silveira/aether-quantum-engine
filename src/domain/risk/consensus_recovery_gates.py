"""Gates que forcam EXPLORE no soft recovery (ACC/live/adapted/chop/Hurst)."""

from __future__ import annotations

from typing import Any

from src.domain.risk.kelly_runtime_config import load_kelly_runtime_from_settings
from src.domain.risk.recovery_conviction import (
    recovery_min_conviction,
    scaled_recovery_min_val_accuracy,
)
from src.domain.risk.stake_sizing import metric_float


def metric_hurst(metrics: dict | None) -> float | None:
    """Le Hurst de metrics top-level, regime_chop ou indicators."""
    if not isinstance(metrics, dict):
        return None
    for key in ("hurst", "regime_chop_hurst"):
        raw = metrics.get(key)
        if raw is None:
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    ind = metrics.get("indicators")
    if isinstance(ind, dict) and ind.get("hurst") is not None:
        try:
            return float(ind["hurst"])
        except (TypeError, ValueError):
            return None
    return None


def acc_below_recovery_floor(metrics: dict | None, consecutive_losses: int) -> bool:
    """True quando val_accuracy live (presente) esta abaixo do piso escalado de recovery."""
    if not isinstance(metrics, dict) or "val_accuracy" not in metrics:
        return False
    try:
        acc = float(metrics.get("val_accuracy"))
    except (TypeError, ValueError):
        return False
    runtime = load_kelly_runtime_from_settings()
    floor = scaled_recovery_min_val_accuracy(
        {"recovery_min_val_accuracy": float(runtime["recovery_min_val_accuracy"])},
        consecutive_losses=int(consecutive_losses),
    )
    return floor > 0.0 and acc + 1e-9 < floor


def live_evidence_blocks_dal(metrics: dict | None, consecutive_losses: int, soft: dict[str, Any]) -> bool:
    """True quando linear alto e live_wr fraco bloqueiam cover DAL (ACC de treino ainda ok)."""
    if not isinstance(metrics, dict) or "live_wr" not in metrics:
        return False
    linear_min = int(soft["live_evidence_force_explore_linear_min"])
    if int(consecutive_losses) < linear_min:
        return False
    try:
        live_n = int(metrics.get("live_n") or 0)
        live_wr = float(metrics["live_wr"])
    except (TypeError, ValueError):
        return False
    n_min = int(soft["live_evidence_force_explore_n_min"])
    wr_max = float(soft["live_evidence_force_explore_wr_max"])
    return live_n >= n_min and live_wr + 1e-12 < wr_max


def adapted_blocks_dal(metrics: dict | None, consecutive_losses: int, soft: dict[str, Any]) -> bool:
    """True quando scale_adapted e linear alto forcam EXPLORE (sem DAL L2+)."""
    if not bool(soft.get("adapted_force_explore", True)):
        return False
    if not isinstance(metrics, dict) or not bool(metrics.get("scale_adapted")):
        return False
    return int(consecutive_losses) >= int(soft["adapted_force_explore_linear_min"])


def conviction_below_recovery_ladder(
    metrics: dict | None,
    consecutive_losses: int,
    kelly_config: dict[str, Any] | None = None,
) -> bool:
    """True quando conviccao do sinal live presente esta abaixo do piso da escada de recovery."""
    if not isinstance(metrics, dict):
        return False
    score = metric_float(metrics, "calibrated_prob", "p_exec", "prob", default=0.0)
    if score <= 0.0 or score >= 1.0:
        return False
    p_side = max(score, 1.0 - score) if score < 0.50 else score
    k_cfg = kelly_config if isinstance(kelly_config, dict) else {}
    min_conv = recovery_min_conviction(
        k_cfg,
        {},
        pending_loss={"_": 1.0},
        consecutive_losses_linear=int(consecutive_losses),
    )
    return p_side + 1e-9 < min_conv


def should_downgrade_recovery_stake(
    metrics: dict | None,
    consecutive_losses: int,
    kelly_config: dict[str, Any] | None = None,
) -> bool:
    """True quando conviccao esta abaixo da escada ou candle fecha contra a ordem em recovery."""
    if not isinstance(metrics, dict):
        return False
    losses = max(0, int(consecutive_losses))
    if conviction_below_recovery_ladder(metrics, losses, kelly_config=kelly_config):
        return True
    if losses >= 1:
        candle = str(metrics.get("closed_micro_candle_dir") or "").strip().upper()
        order_dir = str(metrics.get("exec_direction") or metrics.get("resolved_direction") or "").strip().upper()
        if candle in {"CALL", "PUT"} and order_dir in {"CALL", "PUT"} and candle != order_dir:
            return True
    return False


def resolve_recovery_force_explore(
    metrics: dict | None,
    consecutive_losses: int,
    soft: dict[str, Any],
    *,
    material_pending: bool,
    cover_enabled: bool = True,
) -> tuple[bool, bool, bool, bool, bool]:
    """Resolve flags de explore forcado sob soft recovery (acc, live, adapted, quality, force_early)."""
    acc = acc_below_recovery_floor(metrics, consecutive_losses)
    live = live_evidence_blocks_dal(metrics, consecutive_losses, soft)
    adapted = adapted_blocks_dal(metrics, consecutive_losses, soft)
    cover_disabled_path = bool(material_pending and not cover_enabled)
    if material_pending:
        quality = False
        force_early = cover_disabled_path
    else:
        quality = bool(acc or live or adapted)
        force_early = True
    return acc, live, adapted, quality, force_early
