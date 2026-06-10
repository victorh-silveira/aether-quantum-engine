from types import SimpleNamespace

from src.application.services.execution_direction import recovery_hedge_target
from src.application.services.execution_symbols import (
    candidate_execution_score,
    select_best_execution_candidate,
    select_mandatory_execution_candidate,
)
from src.application.services.execution_symbols_recovery import (
    has_recovery_hedge_candidate,
    inject_recovery_hedge_candidates,
    recovery_blocked_symbols,
    recovery_candidate_pool,
    recovery_rank_score,
)
from src.domain.models.trade import TradeDirection
from tests.market_symbols import ANCHOR, HEDGE_PEER_SYMBOL, PAIR


def test_inject_recovery_hedge_noop_without_loss_context():
    candidates = [(PAIR, TradeDirection.CALL, {"execute": True})]
    out = inject_recovery_hedge_candidates(
        candidates,
        {},
        last_loss_symbol=None,
        last_loss_direction=None,
    )
    assert out == candidates


def test_inject_recovery_hedge_never_adds_opposite_direction():
    candidates = [(PAIR, TradeDirection.CALL, {"execute": True})]
    out = inject_recovery_hedge_candidates(
        candidates,
        {
            HEDGE_PEER_SYMBOL: {
                "direction": TradeDirection.CALL,
                "metrics": {
                    "execute": False,
                    "trade_score": 0.60,
                    "val_accuracy": 0.55,
                    "raw_prob": 0.58,
                },
            }
        },
        last_loss_symbol=PAIR,
        last_loss_direction="CALL",
    )
    assert out == candidates


def test_has_recovery_direction_true_when_same_direction_present():
    assert has_recovery_hedge_candidate(
        [(PAIR, TradeDirection.PUT, {})],
        last_loss_symbol=PAIR,
        last_loss_direction="PUT",
    )


def test_has_recovery_direction_false_when_direction_missing():
    assert not has_recovery_hedge_candidate(
        [(HEDGE_PEER_SYMBOL, TradeDirection.CALL, {})],
        last_loss_symbol=PAIR,
        last_loss_direction="PUT",
    )


def test_recovery_rank_score_bonus_for_matching_direction():
    put_item = (PAIR, TradeDirection.PUT, {"trade_score": 0.55, "val_accuracy": 0.5, "raw_prob": 0.42})
    base_put = candidate_execution_score(put_item[2], recovery_active=True)
    assert recovery_rank_score(put_item, last_loss_direction="PUT", base_score=base_put) >= base_put + 0.06


def test_recovery_rank_score_bonus_for_different_symbol():
    item = (ANCHOR, TradeDirection.CALL, {"trade_score": 0.55, "execute": True, "raw_prob": 0.58})
    base = candidate_execution_score(item[2], recovery_active=True)
    ranked = recovery_rank_score(item, last_loss_symbol=PAIR, last_loss_direction="CALL", base_score=base)
    assert ranked >= base + 0.12


def test_recovery_hedge_target_after_high_side_put_loss():
    target = recovery_hedge_target(PAIR, "PUT")
    assert target == (HEDGE_PEER_SYMBOL, TradeDirection.CALL)


def test_recovery_prefers_opposite_direction_after_loss():
    candidates = [
        (PAIR, TradeDirection.PUT, {"trade_score": 0.70, "val_accuracy": 0.55, "execute": False}),
        (HEDGE_PEER_SYMBOL, TradeDirection.CALL, {"trade_score": 0.72, "val_accuracy": 0.58, "execute": True}),
        (ANCHOR, TradeDirection.PUT, {"trade_score": 0.68, "val_accuracy": 0.56, "execute": True}),
    ]
    best = select_best_execution_candidate(
        candidates,
        last_loss_symbol=PAIR,
        last_loss_direction="PUT",
        diversify_margin=0.08,
        recovery_active=True,
    )
    assert best[0] == HEDGE_PEER_SYMBOL
    assert best[1] == TradeDirection.CALL


def test_select_mandatory_non_recovery_filters_execute_true():
    orch = SimpleNamespace(config={})
    candidates = [
        (ANCHOR, TradeDirection.CALL, {"trade_score": 0.71, "execute": False}),
        (PAIR, TradeDirection.PUT, {"trade_score": 0.55, "execute": True}),
    ]
    best = select_mandatory_execution_candidate(
        orch,
        candidates,
        last_loss_symbol=None,
        diversify_margin=0.08,
        recovery_active=False,
    )
    assert best[0] == PAIR


def test_select_mandatory_recovery_keeps_same_direction_candidate():
    orch = SimpleNamespace(config={})
    candidates = [
        (PAIR, TradeDirection.CALL, {"trade_score": 0.30, "execute": False}),
    ]
    best = select_mandatory_execution_candidate(
        orch,
        candidates,
        last_loss_symbol=PAIR,
        last_loss_direction="CALL",
        diversify_margin=0.08,
        recovery_active=True,
    )
    assert best is not None
    assert best[0] == PAIR
    assert best[1] == TradeDirection.CALL


def test_select_mandatory_returns_none_for_empty_candidates():
    orch = SimpleNamespace(config={})
    assert (
        select_mandatory_execution_candidate(
            orch,
            [],
            last_loss_symbol=None,
            last_loss_direction=None,
            diversify_margin=0.08,
            recovery_active=False,
        )
        is None
    )


def test_select_mandatory_empty_pool_fallback():
    orch = SimpleNamespace(config={})
    candidates = [
        (ANCHOR, TradeDirection.CALL, {"trade_score": 0.70, "execute": False}),
        (PAIR, TradeDirection.CALL, {"trade_score": 0.40, "execute": False}),
    ]
    best = select_mandatory_execution_candidate(
        orch,
        candidates,
        last_loss_symbol=PAIR,
        last_loss_direction="CALL",
        diversify_margin=0.08,
        recovery_active=True,
    )
    assert best is not None
    assert best[0] == ANCHOR


def test_select_mandatory_recovery_prefers_opposite_direction():
    orch = SimpleNamespace(config={})
    candidates = [
        (HEDGE_PEER_SYMBOL, TradeDirection.PUT, {"trade_score": 0.55, "val_accuracy": 0.60, "execute": True}),
        (ANCHOR, TradeDirection.CALL, {"trade_score": 0.62, "val_accuracy": 0.58, "execute": True}),
        (PAIR, TradeDirection.CALL, {"trade_score": 0.40, "execute": False}),
    ]
    best = select_mandatory_execution_candidate(
        orch,
        candidates,
        last_loss_symbol=PAIR,
        last_loss_direction="CALL",
        diversify_margin=0.08,
        recovery_active=True,
    )
    assert best[0] == HEDGE_PEER_SYMBOL
    assert best[1] == TradeDirection.PUT


def test_recovery_candidate_pool_keeps_opposite_direction_candidates():
    candidates = [
        (PAIR, TradeDirection.PUT, {"execute": True}),
        (HEDGE_PEER_SYMBOL, TradeDirection.CALL, {"execute": True}),
    ]
    result = recovery_candidate_pool(
        candidates,
        last_loss_symbol=PAIR,
        last_loss_direction="PUT",
        recovery_active=True,
    )
    assert len(result) == 2


def test_recovery_candidate_pool_without_direction_filter_when_inactive():
    candidates = [(PAIR, TradeDirection.PUT, {"execute": True})]
    result = recovery_candidate_pool(
        candidates,
        last_loss_symbol=PAIR,
        last_loss_direction=None,
        recovery_active=False,
    )
    assert result == candidates


def test_recovery_rank_score_call_raw_bonus():
    item = (PAIR, TradeDirection.CALL, {"trade_score": 0.55, "val_accuracy": 0.5, "raw_prob": 0.58})
    base = candidate_execution_score(item[2], recovery_active=True)
    assert recovery_rank_score(item, last_loss_direction="CALL", base_score=base) >= base + 0.08


def test_recovery_rank_score_put_raw_bonus():
    item = (PAIR, TradeDirection.PUT, {"trade_score": 0.55, "val_accuracy": 0.5, "raw_prob": 0.42})
    base = candidate_execution_score(item[2], recovery_active=True)
    assert recovery_rank_score(item, last_loss_direction="PUT", base_score=base) >= base + 0.08


def test_recovery_blocked_symbols_excludes_streak_and_cooldown():
    rm = SimpleNamespace(
        recovery_symbol_loss_streak={PAIR: 2},
        symbol_loss_cooldown={HEDGE_PEER_SYMBOL: 1},
    )
    blocked = recovery_blocked_symbols(rm, {"recovery_martingale_max_losses_per_symbol": 2})
    assert blocked == frozenset({PAIR, HEDGE_PEER_SYMBOL})


def test_recovery_candidate_pool_skips_blocked_symbols():
    candidates = [
        (PAIR, TradeDirection.CALL, {"execute": True, "trade_score": 0.60}),
        (HEDGE_PEER_SYMBOL, TradeDirection.CALL, {"execute": True, "trade_score": 0.55}),
    ]
    result = recovery_candidate_pool(
        candidates,
        last_loss_symbol=None,
        last_loss_direction="CALL",
        recovery_active=True,
        skip_symbols=frozenset({PAIR}),
    )
    assert len(result) == 1
    assert result[0][0] == HEDGE_PEER_SYMBOL


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
