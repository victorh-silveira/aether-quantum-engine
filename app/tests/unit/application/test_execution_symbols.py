from types import SimpleNamespace

from src.application.services.execution_symbols import (
    candidate_execution_score,
    format_execution_alternates,
    pending_recovery_active,
    select_best_execution_candidate,
    select_mandatory_execution_candidate,
    symbols_eligible_for_execution,
)
from src.domain.models.trade import TradeDirection
from tests.market_symbols import ANCHOR, PAIR


def test_symbols_eligible_for_execution():
    symbols = [ANCHOR, PAIR]
    assert symbols_eligible_for_execution(ANCHOR, symbols, include_anchor=False) == [PAIR]
    assert symbols_eligible_for_execution(ANCHOR, symbols, include_anchor=True) == symbols


def test_format_execution_alternates_excludes_selected():
    candidates = [
        (ANCHOR, TradeDirection.PUT, {"trade_score": 0.80, "raw_prob": 0.80, "val_accuracy": 0.60, "execute": True}),
        (PAIR, TradeDirection.CALL, {"trade_score": 0.78, "raw_prob": 0.78, "val_accuracy": 0.50, "execute": True}),
    ]
    assert format_execution_alternates(candidates, exclude_symbol=ANCHOR) == f"{PAIR}(0.78)"


def test_pending_recovery_active():
    assert pending_recovery_active({}) is False
    assert pending_recovery_active({ANCHOR: 100.0}) is True


def test_select_best_execution_candidate_diversify_margin_picks_alt():
    candidates = [
        (ANCHOR, TradeDirection.PUT, {"trade_score": 0.80, "raw_prob": 0.80, "val_accuracy": 0.50, "execute": True}),
        (PAIR, TradeDirection.CALL, {"trade_score": 0.79, "raw_prob": 0.79, "val_accuracy": 0.48, "execute": True}),
    ]
    best = select_best_execution_candidate(
        candidates,
        last_loss_symbol=ANCHOR,
        diversify_margin=0.05,
        recovery_active=False,
    )
    assert best[0] == PAIR


def test_candidate_execution_score_recovery_weights_val_accuracy():
    metrics = {"trade_score": 0.80, "raw_prob": 0.80, "val_accuracy": 0.40, "execute": True}
    normal = candidate_execution_score(metrics, recovery_active=False, symbol="R_50")
    recovery = candidate_execution_score(
        metrics,
        recovery_active=True,
        symbol="R_50",
        exec_direction=TradeDirection.CALL,
        last_loss_symbol="R_10",
        last_loss_direction="CALL",
    )
    high_val = candidate_execution_score(
        {"trade_score": 0.80, "raw_prob": 0.80, "val_accuracy": 0.60, "execute": True},
        recovery_active=True,
        symbol="R_50",
        exec_direction=TradeDirection.CALL,
        last_loss_symbol="R_10",
        last_loss_direction="CALL",
    )
    assert high_val > recovery
    assert recovery < normal


def test_select_mandatory_falls_back_when_pool_empty():
    orch = SimpleNamespace(config={})
    candidates = [
        (ANCHOR, TradeDirection.CALL, {"trade_score": 0.80, "execute": False}),
        (PAIR, TradeDirection.PUT, {"trade_score": 0.43, "execute": False}),
    ]
    best = select_mandatory_execution_candidate(
        orch,
        candidates,
        last_loss_symbol=None,
        diversify_margin=0.08,
        recovery_active=False,
    )
    assert best is not None
    assert best[0] == ANCHOR
