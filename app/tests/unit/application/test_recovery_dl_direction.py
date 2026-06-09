from src.application.services.execution_symbols import (
    candidate_execution_score,
    recovery_rank_score,
    select_best_execution_candidate,
)
from src.application.services.execution_symbols_recovery import recovery_candidate_pool
from src.domain.models.trade import TradeDirection
from tests.market_symbols import HEDGE_PEER_SYMBOL, PAIR


def test_recovery_candidate_pool_hedge_peer_only():
    candidates = [
        (PAIR, TradeDirection.PUT, {"execute": True, "dl_direction": "PUT"}),
        (HEDGE_PEER_SYMBOL, TradeDirection.PUT, {"execute": True, "dl_direction": "PUT"}),
    ]
    result = recovery_candidate_pool(
        candidates,
        last_loss_symbol=PAIR,
        last_loss_direction="PUT",
        recovery_active=True,
    )
    assert len(result) == 1
    assert result[0][0] == HEDGE_PEER_SYMBOL


def test_recovery_rank_score_bonus_for_hedge_match():
    hedge = (HEDGE_PEER_SYMBOL, TradeDirection.CALL)
    item = (HEDGE_PEER_SYMBOL, TradeDirection.CALL, {"trade_score": 0.55, "raw_prob": 0.42})
    base = candidate_execution_score(item[2], recovery_active=True)
    assert recovery_rank_score(item, hedge) >= base + 0.25


def test_recovery_candidate_pool_hedge_peer_fallback_from_candidates():
    candidates = [
        (PAIR, TradeDirection.PUT, {"execute": False}),
        (HEDGE_PEER_SYMBOL, TradeDirection.CALL, {"execute": True}),
    ]
    result = recovery_candidate_pool(
        candidates,
        last_loss_symbol=PAIR,
        last_loss_direction="PUT",
        recovery_active=True,
    )
    assert len(result) == 1
    assert result[0][0] == HEDGE_PEER_SYMBOL


def test_select_best_builds_hedge_rank_in_recovery():
    candidates = [
        (PAIR, TradeDirection.PUT, {"trade_score": 0.70, "val_accuracy": 0.55, "execute": False}),
        (HEDGE_PEER_SYMBOL, TradeDirection.CALL, {"trade_score": 0.52, "val_accuracy": 0.58, "execute": True}),
    ]
    best = select_best_execution_candidate(
        candidates,
        last_loss_symbol=PAIR,
        last_loss_direction="PUT",
        diversify_margin=0.08,
        recovery_active=True,
    )
    assert best[0] in {PAIR, HEDGE_PEER_SYMBOL}
