from types import SimpleNamespace

from src.application.services.execution_symbols import select_best_execution_candidate
from src.application.services.execution_symbols_recovery import (
    recovery_blocked_symbols,
    recovery_candidate_pool,
)
from src.domain.models.trade import TradeDirection
from tests.market_symbols import ALT_SYMBOL, ANCHOR, HEDGE_PEER_SYMBOL, PAIR


def test_recovery_blocked_symbols_never_excludes():
    rm = SimpleNamespace(
        consecutive_losses_linear=0,
        last_loss_symbol="R_10",
    )
    blocked = recovery_blocked_symbols(rm, {})
    assert blocked == frozenset()


def test_recovery_blocked_symbols_rotates_after_linear_loss():
    rm = SimpleNamespace(
        consecutive_losses_linear=1,
        last_loss_symbol="R_10",
    )
    blocked = recovery_blocked_symbols(rm, {"symbol_loss_rotation_cycles": 1})
    assert blocked == frozenset({"R_10"})


def test_recovery_blocked_symbols_respects_rotation_cycles():
    rm = SimpleNamespace(
        consecutive_losses_linear=1,
        last_loss_symbol="R_10",
    )
    blocked = recovery_blocked_symbols(rm, {"symbol_loss_rotation_cycles": 2})
    assert blocked == frozenset()


def test_recovery_candidate_pool_skips_blocked_symbols():
    candidates = [
        (ALT_SYMBOL, TradeDirection.CALL, {"execute": True, "trade_score": 0.60}),
        (ANCHOR, TradeDirection.CALL, {"execute": True, "trade_score": 0.55}),
    ]
    result = recovery_candidate_pool(
        candidates,
        last_loss_symbol=None,
        last_loss_direction="CALL",
        recovery_active=True,
        skip_symbols=frozenset({ALT_SYMBOL}),
    )
    assert len(result) == 1
    assert result[0][0] == ANCHOR


def test_select_best_returns_none_for_empty_pool():
    assert (
        select_best_execution_candidate(
            [],
            last_loss_symbol=None,
            diversify_margin=0.08,
            recovery_active=False,
        )
        is None
    )


def test_recovery_candidate_pool_keeps_single_opposite_direction():
    candidates = [
        (HEDGE_PEER_SYMBOL, TradeDirection.CALL, {"execute": True}),
    ]
    result = recovery_candidate_pool(
        candidates,
        last_loss_symbol=PAIR,
        last_loss_direction="PUT",
        recovery_active=True,
    )
    assert len(result) == 1
    assert result[0][1] == TradeDirection.CALL
