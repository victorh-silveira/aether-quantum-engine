from src.application.services.execution_symbols import (
    candidate_execution_score,
    select_best_execution_candidate,
)
from src.application.services.execution_symbols_recovery import (
    recovery_candidate_pool,
    recovery_rank_score,
)
from src.domain.models.trade import TradeDirection
from tests.market_symbols import ANCHOR, HEDGE_PEER_SYMBOL, PAIR


def test_recovery_candidate_pool_keeps_all_directions():
    candidates = [
        (PAIR, TradeDirection.PUT, {"execute": True, "dl_direction": "PUT"}),
        (ANCHOR, TradeDirection.PUT, {"execute": True, "dl_direction": "PUT"}),
        (HEDGE_PEER_SYMBOL, TradeDirection.CALL, {"execute": True, "dl_direction": "CALL"}),
    ]
    result = recovery_candidate_pool(
        candidates,
        last_loss_symbol=PAIR,
        last_loss_direction="PUT",
        recovery_active=True,
    )
    assert len(result) == 3


def test_recovery_rank_score_penalizes_same_direction():
    item = ("SYM", TradeDirection.PUT, {"trade_score": 0.55, "raw_prob": 0.42})
    base = candidate_execution_score(item[2], recovery_active=True, symbol="SYM")
    assert recovery_rank_score(item, last_loss_direction="PUT", base_score=base) <= base


def test_recovery_select_prefers_different_symbol_after_put_loss():
    candidates = [
        (PAIR, TradeDirection.PUT, {"execute": False}),
        (ANCHOR, TradeDirection.PUT, {"execute": True, "trade_score": 0.65, "val_accuracy": 0.55, "raw_prob": 0.45}),
        (HEDGE_PEER_SYMBOL, TradeDirection.CALL, {"execute": True}),
    ]
    best = select_best_execution_candidate(
        candidates,
        last_loss_symbol=PAIR,
        last_loss_direction="PUT",
        diversify_margin=0.08,
        recovery_active=True,
    )
    assert best is not None
    assert best[0] == ANCHOR
    assert best[1] == TradeDirection.PUT


def test_select_best_can_pick_opposite_direction_when_stronger():
    candidates = [
        (PAIR, TradeDirection.CALL, {"trade_score": 0.45, "val_accuracy": 0.55, "execute": True, "raw_prob": 0.58}),
        (ANCHOR, TradeDirection.PUT, {"trade_score": 0.64, "val_accuracy": 0.80, "execute": True, "raw_prob": 0.48}),
    ]
    best = select_best_execution_candidate(
        candidates,
        last_loss_symbol=PAIR,
        last_loss_direction="CALL",
        diversify_margin=0.08,
        recovery_active=True,
    )
    assert best is not None
    assert best[0] == ANCHOR
    assert best[1] == TradeDirection.PUT
