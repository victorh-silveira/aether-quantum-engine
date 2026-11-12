"""Stake do pullback com base de 1% e recuperacao pelo payout cotado."""

from __future__ import annotations

import math
from typing import Any

from src.domain.risk.stop_win_target import resolve_stop_win_target


def resolve_pullback_amort_steps(loss_amount: float, quote_rate: float, cap_stake: float) -> int:
    """Calcula etapas para recovery suave sem exceder o teto por ordem."""
    if not all(math.isfinite(value) for value in (loss_amount, quote_rate, cap_stake)):
        return 1
    if loss_amount <= 0.0 or quote_rate <= 0.0 or cap_stake <= 0.0:
        return 1
    needed = loss_amount / quote_rate
    return max(1, math.ceil(needed / cap_stake)) if needed > cap_stake else 1


def resolve_pullback_single_strike_target(
    pb_cfg: dict, rm_cfg: dict, risk_manager: Any, fallback_bankroll: float = 0.0
) -> float:
    """Calcula o alvo liquido restante do STOP WIN quando single_strike esta ativo."""
    if not pb_cfg.get("single_strike", False):
        return 0.0
    init_b = float(getattr(risk_manager, "initial_bankroll", 0.0) or fallback_bankroll)
    sw = resolve_stop_win_target(rm_cfg, init_b)
    pnl = float(getattr(risk_manager, "total_session_profit", 0.0) or 0.0)
    return max(0.0, sw - pnl) if sw > 0.0 else 0.0


def resolve_pullback_order_params(
    config: dict,
    risk_manager: Any,
    available: float,
    unres_loss: float,
    quote_rate: float,
    fallback_bankroll: float = 0.0,
) -> tuple[float, float, int, float]:
    """Resolve stake minimo, alvo single_strike, etapas de recovery e teto por ordem."""
    pb_cfg = config.get("pullback_strategy", {})
    rm_cfg = config.get("risk_management", {})
    minimum = float(rm_cfg.get("params", {}).get("stake_min", 1.0))
    sst = resolve_pullback_single_strike_target(pb_cfg, rm_cfg, risk_manager, fallback_bankroll)
    cap_limit = math.floor(available * 0.05 * 100) / 100
    steps = (
        resolve_pullback_amort_steps(unres_loss, quote_rate, cap_limit)
        if pb_cfg.get("smooth_recovery", False)
        else int(pb_cfg.get("amort_steps", 1))
    )
    return minimum, sst, steps, cap_limit


def resolve_pullback_prequote_stake(
    config: dict,
    risk_manager: Any,
    balance: float,
    min_rate: float,
    *,
    last_rate: float = 0.0,
    unreserved: float = 0.0,
) -> float:
    """Estima o stake alvo de prequote alinhado a single_strike e recuperacao."""
    est_rate = last_rate if last_rate > 0 else max(min_rate, 0.01)
    minimum, sst, steps, _ = resolve_pullback_order_params(config, risk_manager, balance, unreserved, est_rate, balance)
    stake, _ = pullback_stake(balance, unreserved, est_rate, minimum, single_strike_target=sst, amort_steps=steps)
    return stake


def pullback_stake(
    available: float,
    pending: float,
    payout_rate: float,
    stake_min: float,
    single_strike_target: float = 0.0,
    amort_steps: int = 1,
) -> tuple[float, float]:
    """Devolve stake e parcela da perda que uma vitoria poderia amortizar."""
    if not all(math.isfinite(value) for value in (available, pending, payout_rate, stake_min, single_strike_target)):
        return 0.0, 0.0
    if available <= 0 or payout_rate <= 0 or stake_min <= 0:
        return 0.0, 0.0
    cap = math.floor(min(available, available * 0.05) * 100) / 100
    if cap + 1e-9 < stake_min:
        return 0.0, 0.0
    base = math.floor(available * 0.01 * 100) / 100
    if pending > 0:
        steps = max(1, int(amort_steps))
        target = max(base, stake_min, (pending / steps) / payout_rate)
    elif single_strike_target > 0:
        target = max(base, stake_min, single_strike_target / payout_rate)
    else:
        target = max(base, stake_min)
    stake = min(cap, math.ceil(target * 100 - 1e-9) / 100)
    covered = min(max(0.0, pending), stake * payout_rate)
    return stake, covered
