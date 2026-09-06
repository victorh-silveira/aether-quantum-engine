"""Leitura de soft/hard de gate para sizing Kelly (domain puro)."""

from __future__ import annotations

from typing import Any


_SOFT_SIGNAL_FLAGS = (
    "loss_clf_soft",
    "cal_margin_soft",
    "neg_edge_soft",
    "mini_pair_soft",
    "regime_chop_soft",
    "fusion_weak_ev_soft",
    "anti_loss_soft",
    "micro_discord_follow_soft",
)


def blocks_single_strike_boost(metrics: dict[str, Any] | None) -> bool:
    """True se SOFT_SIZE ou flag soft de sinal — nao aplicar boost Single-Strike."""
    if not isinstance(metrics, dict):
        return False
    if str(metrics.get("gate_verdict") or "").strip().upper() == "SOFT_SIZE":
        return True
    return any(bool(metrics.get(flag)) for flag in _SOFT_SIGNAL_FLAGS)
