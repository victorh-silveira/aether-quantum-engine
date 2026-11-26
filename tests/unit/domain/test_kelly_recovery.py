"""Invariantes financeiras de Kelly, pisos e recuperacao por lucro realizado."""

import math

import pytest
from hypothesis import (
    given,
    strategies as st,
)

from src.domain.risk.kelly_recovery import size_kelly_recovery, sizing_from_config
from src.domain.risk.risk_manager import RiskManager


def test_kelly_floor_positive_and_recovery():
    floor = size_kelly_recovery(1000, 0.50, 0.8)
    assert floor.stake == 10 and floor.source == "FLOOR" and floor.kelly_raw < 0
    positive = size_kelly_recovery(1000, 0.60, 0.8)
    assert positive.stake == 25 and positive.source == "KELLY"
    recovery = size_kelly_recovery(1000, 0.5, 0.8, 100)
    assert recovery.recovery_steps == 3 and recovery.stake == 41.67
    assert recovery.source == "RECOVERY" and recovery.capped
    assert recovery.recovery_slice == pytest.approx(100 / 3)
    assert size_kelly_recovery(1000, 1, 0.8).stake == 50
    disabled = size_kelly_recovery(1000, 0.5, 0.8, 100, recovery_enabled=False)
    assert disabled.stake == 10 and disabled.recovery_slice == 0
    assert sizing_from_config(100, 0.5, 0.8, 0, {}).stake == 1


@pytest.mark.parametrize(
    "args,kwargs",
    [
        ((math.nan, 0.5, 0.8), {}),
        ((0, 0.5, 0.8), {}),
        ((100, 1.2, 0.8), {}),
        ((100, 0.5, 0), {}),
        ((100, 0.5, 0.8, -1), {}),
        ((100, 0.5, 0.8), {"max_pct": 0.06}),
        ((100, 0.5, 0.8), {"fraction": 0}),
        ((100, 0.5, 0.8), {"stake_min": 0.1}),
        ((10, 0.5, 0.8), {}),
    ],
)
def test_invalid_financial_inputs_never_create_order(args, kwargs):
    with pytest.raises(ValueError):
        size_kelly_recovery(*args, **kwargs)


@given(
    balance=st.floats(min_value=100, max_value=1e6, allow_nan=False),
    prob=st.floats(min_value=0, max_value=1, allow_nan=False),
    payout=st.floats(min_value=0.1, max_value=2, allow_nan=False),
    pending=st.floats(min_value=0, max_value=1e7, allow_nan=False),
)
def test_stake_always_respects_cap_and_floor(balance, prob, payout, pending):
    sized = size_kelly_recovery(balance, prob, payout, pending)
    assert balance * 0.01 - 1e-8 <= sized.stake <= balance * 0.05 + 1e-8
    assert sized.stake <= balance
    assert abs(sized.stake * 100 - round(sized.stake * 100)) < 1e-6


def test_pending_losses_partial_wins_duplicates_restore_and_reset():
    risk = RiskManager()
    risk.on_trade_settled(1, -100)
    risk.on_trade_settled(2, 30)
    assert risk.pending_loss == 70 and risk.recovery_losses == 1
    payload = risk.get_state()
    restored = RiskManager()
    restored.restore_state(payload)
    assert restored.on_trade_settled(1, -100) == 0
    assert restored.pending_loss == 70
    restored.on_trade_settled(3, 80)
    assert restored.pending_loss == 0 and restored.recovery_losses == 0
    restored.on_trade_settled(4, -10)
    restored.reset_session(1000)
    assert restored.pending_loss == 0 and restored.recovery_losses == 0
    restored.restore_state({"total_session_profit": -30})
    assert restored.pending_loss == 0
