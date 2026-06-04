from src.application.services.execution_symbols import select_best_execution_candidate
from src.domain.models.trade import TradeDirection
from tests.market_symbols import ANCHOR, PAIR


def test_select_best_candidate_keeps_top_when_clear_lead():
    candidates = [
        (ANCHOR, TradeDirection.PUT, {"conviction": 0.81, "raw_conviction": 0.82, "val_accuracy": 0.50}),
        (PAIR, TradeDirection.CALL, {"conviction": 0.64, "raw_conviction": 0.60, "val_accuracy": 0.45}),
    ]
    best = select_best_execution_candidate(
        candidates,
        last_loss_symbol=ANCHOR,
        diversify_margin=0.06,
        recovery_active=False,
    )
    assert best[0] == ANCHOR


def test_select_best_candidate_diversifies_after_loss():
    candidates = [
        (ANCHOR, TradeDirection.PUT, {"conviction": 0.67, "raw_conviction": 0.67, "val_accuracy": 0.40}),
        (PAIR, TradeDirection.CALL, {"conviction": 0.62, "raw_conviction": 0.64, "val_accuracy": 0.50}),
    ]
    best = select_best_execution_candidate(
        candidates,
        last_loss_symbol=ANCHOR,
        diversify_margin=0.10,
        recovery_active=True,
    )
    assert best[0] == PAIR


def test_select_best_candidate_prefers_high_val_in_recovery():
    candidates = [
        (
            PAIR,
            TradeDirection.PUT,
            {"trade_score": 0.57, "conviction": 0.57, "val_accuracy": 0.60, "edge": 0.07},
        ),
        (
            ANCHOR,
            TradeDirection.CALL,
            {"trade_score": 0.59, "conviction": 0.59, "val_accuracy": 0.42, "edge": 0.09},
        ),
    ]
    best = select_best_execution_candidate(
        candidates,
        last_loss_symbol=None,
        diversify_margin=0.10,
        recovery_active=True,
    )
    assert best[0] == PAIR


def test_select_best_candidate_single_candidate():
    candidates = [(ANCHOR, TradeDirection.PUT, {"conviction": 0.70, "raw_conviction": 0.70})]
    assert (
        select_best_execution_candidate(
            candidates,
            last_loss_symbol=ANCHOR,
            diversify_margin=0.06,
            recovery_active=False,
        )[0]
        == ANCHOR
    )
