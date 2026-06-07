from types import SimpleNamespace
from unittest.mock import patch

from src.application.services.execution_direction import recovery_hedge_target
from src.application.services.execution_symbols import (
    candidate_execution_score,
    recovery_rank_score,
    select_best_execution_candidate,
    select_mandatory_execution_candidate,
)
from src.application.services.execution_symbols_recovery import (
    has_recovery_hedge_candidate,
    inject_recovery_hedge_candidates,
    recovery_candidate_pool,
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


def test_inject_recovery_hedge_skips_when_candidate_already_present():
    candidates = [(HEDGE_PEER_SYMBOL, TradeDirection.CALL, {"execute": True})]
    out = inject_recovery_hedge_candidates(
        candidates,
        {HEDGE_PEER_SYMBOL: {"direction": TradeDirection.CALL, "metrics": {}}},
        last_loss_symbol=PAIR,
        last_loss_direction="CALL",
    )
    assert out == candidates


def test_inject_recovery_hedge_skips_when_forced_build_fails():
    candidates = [(PAIR, TradeDirection.CALL, {"execute": False})]
    out = inject_recovery_hedge_candidates(
        candidates,
        {HEDGE_PEER_SYMBOL: {"direction": None, "metrics": {"execute": False}}},
        last_loss_symbol=PAIR,
        last_loss_direction="CALL",
    )
    assert out == candidates


def test_has_recovery_hedge_true_when_hedge_not_applicable():
    assert has_recovery_hedge_candidate(
        [(ANCHOR, TradeDirection.CALL, {})],
        last_loss_symbol=None,
        last_loss_direction=None,
    )


def test_recovery_rank_score_call_and_put_raw_bonus():
    hedge = recovery_hedge_target(PAIR, "PUT")
    call_item = (HEDGE_PEER_SYMBOL, TradeDirection.CALL, {"trade_score": 0.55, "val_accuracy": 0.5, "raw_prob": 0.58})
    put_item = (PAIR, TradeDirection.PUT, {"trade_score": 0.55, "val_accuracy": 0.5, "raw_prob": 0.42})
    call_score = recovery_rank_score(call_item, hedge)
    put_score = recovery_rank_score(put_item, hedge)
    base_call = candidate_execution_score(call_item[2], recovery_active=True)
    base_put = candidate_execution_score(put_item[2], recovery_active=True)
    assert call_score >= base_call + 0.26
    assert put_score >= base_put + 0.01


def test_recovery_hedge_target_after_high_side_put_loss():
    target = recovery_hedge_target(PAIR, "PUT")
    assert target == (HEDGE_PEER_SYMBOL, TradeDirection.CALL)


def test_inject_recovery_hedge_adds_put_on_peer_after_high_side_call_loss():
    decisions = {
        HEDGE_PEER_SYMBOL: {"direction": TradeDirection.CALL, "metrics": {"trade_score": 0.55, "execute": True}},
        PAIR: {"direction": TradeDirection.CALL, "metrics": {"trade_score": 0.44, "execute": False}},
    }
    candidates = [
        (PAIR, TradeDirection.CALL, {"trade_score": 0.44, "execute": False}),
    ]
    expanded = inject_recovery_hedge_candidates(
        candidates,
        decisions,
        last_loss_symbol=PAIR,
        last_loss_direction="CALL",
    )
    assert has_recovery_hedge_candidate(expanded, last_loss_symbol=PAIR, last_loss_direction="CALL")
    hedged = [item for item in expanded if item[0] == HEDGE_PEER_SYMBOL and item[1] == TradeDirection.CALL]
    assert len(hedged) == 1


def test_recovery_selects_hedge_symbol_and_direction():
    candidates = [
        (PAIR, TradeDirection.PUT, {"trade_score": 0.70, "val_accuracy": 0.55, "execute": False}),
        (HEDGE_PEER_SYMBOL, TradeDirection.PUT, {"trade_score": 0.55, "val_accuracy": 0.60, "execute": True}),
        (HEDGE_PEER_SYMBOL, TradeDirection.CALL, {"trade_score": 0.52, "val_accuracy": 0.58, "execute": True}),
    ]
    best = select_best_execution_candidate(
        candidates,
        last_loss_symbol=PAIR,
        last_loss_direction="PUT",
        diversify_margin=0.08,
        recovery_active=True,
    )
    assert best[0] == HEDGE_PEER_SYMBOL
    assert best[1] == TradeDirection.PUT


def test_select_mandatory_non_recovery_filters_execute_true():
    orch = SimpleNamespace(config={"deep_learning": {"post_loss_flip_raw_min": 0.62}})
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
        flip_raw_min=0.62,
    )
    assert best[0] == PAIR


def test_select_mandatory_recovery_restores_pool_when_narrowed_empty():
    orch = SimpleNamespace(config={"deep_learning": {"post_loss_flip_raw_min": 0.99}})
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
        flip_raw_min=0.99,
    )
    assert best[0] == PAIR


def test_select_mandatory_returns_first_candidate_when_pool_empty():
    orch = SimpleNamespace(config={"deep_learning": {"post_loss_flip_raw_min": 0.99}})
    candidates = [(ANCHOR, TradeDirection.CALL, {"execute": False})]
    with (
        patch(
            "src.application.services.execution_symbols.recovery_candidate_pool",
            return_value=[],
        ),
        patch(
            "src.application.services.execution_symbols.list",
            side_effect=lambda seq: [] if seq is candidates else list(seq),
        ),
    ):
        best = select_mandatory_execution_candidate(
            orch,
            candidates,
            last_loss_symbol=PAIR,
            last_loss_direction="CALL",
            diversify_margin=0.08,
            recovery_active=True,
            flip_raw_min=0.99,
        )
    assert best == candidates[0]


def test_select_mandatory_prefers_execute_true_when_available():
    orch = SimpleNamespace(config={"deep_learning": {"post_loss_flip_raw_min": 0.62}})
    candidates = [
        (ANCHOR, TradeDirection.CALL, {"trade_score": 0.71, "execute": False, "raw_prob": 0.56}),
        (PAIR, TradeDirection.PUT, {"trade_score": 0.55, "execute": True, "raw_prob": 0.55}),
    ]
    best = select_mandatory_execution_candidate(
        orch,
        candidates,
        last_loss_symbol=None,
        diversify_margin=0.08,
        recovery_active=False,
        flip_raw_min=0.62,
    )
    assert best[0] == PAIR


def test_recovery_candidate_pool_fallback_to_original_candidates():
    candidates = [
        (PAIR, TradeDirection.PUT, {"execute": True}),
        (HEDGE_PEER_SYMBOL, TradeDirection.CALL, {"execute": False}),
    ]
    result = recovery_candidate_pool(
        candidates,
        last_loss_symbol=PAIR,
        last_loss_direction="PUT",
        recovery_active=True,
    )
    assert len(result) == 1
    assert result[0][0] == HEDGE_PEER_SYMBOL


def test_inject_recovery_hedge_candidates_no_entry():
    candidates = [(PAIR, TradeDirection.CALL, {"execute": True})]
    out = inject_recovery_hedge_candidates(
        candidates,
        {},
        last_loss_symbol=PAIR,
        last_loss_direction="CALL",
    )
    assert out == candidates


def test_has_recovery_hedge_candidate_no_last_loss():
    assert has_recovery_hedge_candidate(
        [(PAIR, TradeDirection.CALL, {})],
        last_loss_symbol=None,
        last_loss_direction=None,
    )


def test_has_recovery_hedge_candidate_no_peer():
    assert (
        has_recovery_hedge_candidate(
            [],
            last_loss_symbol="R_50",
            last_loss_direction="CALL",
        )
        is True
    )


def test_inject_recovery_hedge_skips_when_peer_execute_false():
    candidates = [(PAIR, TradeDirection.CALL, {"execute": True})]
    out = inject_recovery_hedge_candidates(
        candidates,
        {HEDGE_PEER_SYMBOL: {"direction": TradeDirection.CALL, "metrics": {"execute": False}}},
        last_loss_symbol=PAIR,
        last_loss_direction="CALL",
    )
    assert out == candidates
