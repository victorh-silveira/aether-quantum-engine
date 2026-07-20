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
from tests.market_symbols import ALT_SYMBOL, ANCHOR


def test_symbols_eligible_for_execution():
    symbols = [ANCHOR, ALT_SYMBOL]
    assert symbols_eligible_for_execution(ANCHOR, symbols, include_anchor=False) == [ALT_SYMBOL]
    assert symbols_eligible_for_execution(ANCHOR, symbols, include_anchor=True) == symbols


def test_format_execution_alternates_excludes_selected():
    candidates = [
        (ANCHOR, TradeDirection.PUT, {"trade_score": 0.80, "raw_prob": 0.80, "val_accuracy": 0.60, "execute": True}),
        (
            ALT_SYMBOL,
            TradeDirection.CALL,
            {"trade_score": 0.78, "raw_prob": 0.78, "val_accuracy": 0.50, "execute": True},
        ),
    ]
    assert format_execution_alternates(candidates, exclude_symbol=ANCHOR) == f"{ALT_SYMBOL}(0.78)"


def test_pending_recovery_active():
    assert pending_recovery_active({}) is False
    assert pending_recovery_active({ANCHOR: 100.0}) is True


def test_select_best_execution_candidate_picks_highest_score():
    candidates = [
        (ANCHOR, TradeDirection.PUT, {"trade_score": 0.80, "raw_prob": 0.80, "val_accuracy": 0.50, "execute": True}),
        (
            ALT_SYMBOL,
            TradeDirection.CALL,
            {"trade_score": 0.79, "raw_prob": 0.79, "val_accuracy": 0.48, "execute": True},
        ),
    ]
    best = select_best_execution_candidate(
        candidates,
        last_loss_symbol=None,
        recovery_active=False,
    )
    assert best[0] == ANCHOR


def test_candidate_execution_score_recovery_weights_val_accuracy():
    metrics = {"trade_score": 0.80, "raw_prob": 0.80, "val_accuracy": 0.40, "execute": True}
    normal = candidate_execution_score(metrics, recovery_active=False, symbol="R_10")
    recovery = candidate_execution_score(
        metrics,
        recovery_active=True,
        symbol="R_10",
        exec_direction=TradeDirection.CALL,
        last_loss_symbol="R_10",
    )
    high_val = candidate_execution_score(
        {"trade_score": 0.80, "raw_prob": 0.80, "val_accuracy": 0.60, "execute": True},
        recovery_active=True,
        symbol="R_10",
        exec_direction=TradeDirection.CALL,
        last_loss_symbol="R_10",
    )
    assert high_val > recovery
    assert recovery < normal


def test_candidate_execution_score_uses_exec_direction_from_metrics():
    metrics = {
        "trade_score": 0.80,
        "raw_prob": 0.80,
        "val_accuracy": 0.60,
        "execute": True,
        "exec_direction": "PUT",
    }
    score = candidate_execution_score(metrics, recovery_active=False, symbol="R_10")
    assert score > 0.0


def test_select_mandatory_falls_back_when_pool_empty():
    orch = SimpleNamespace(config={})
    candidates = [
        (ANCHOR, TradeDirection.CALL, {"trade_score": 0.80, "execute": False}),
        (ALT_SYMBOL, TradeDirection.PUT, {"trade_score": 0.43, "execute": False}),
    ]
    best = select_mandatory_execution_candidate(
        orch,
        candidates,
        last_loss_symbol=None,
        recovery_active=False,
    )
    assert best is not None
    assert best[0] == ANCHOR
