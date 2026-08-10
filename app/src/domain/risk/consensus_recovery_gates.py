"""Gates que forcam EXPLORE no soft recovery (ACC/live/adapted/chop/Hurst)."""

from __future__ import annotations

from typing import Any

from src.domain.risk.kelly_runtime_config import load_kelly_runtime_from_settings
from src.domain.risk.recovery_conviction import scaled_recovery_min_val_accuracy


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


def chop_neg_edge_dampens_dal(metrics: dict | None) -> bool:
    """True quando NEG_EDGE soft/hard pede EXPLORE em vez de DAL agressivo."""
    if not isinstance(metrics, dict):
        return False
    if bool(metrics.get("neg_edge_soft")):
        return True
    if str(metrics.get("gate_reason") or "").strip() == "neg_edge":
        return True
    if str(metrics.get("signal_status") or "").strip().upper() == "SKIP:NEG_EDGE":
        return True
    return str(metrics.get("signal_skip_waived") or "").strip() == "neg_edge_soft"


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
