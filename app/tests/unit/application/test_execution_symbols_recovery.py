from types import SimpleNamespace

from src.application.services.execution_symbols import (
    candidate_execution_score,
    select_best_execution_candidate,
    select_mandatory_execution_candidate,
)
from src.application.services.execution_symbols_recovery import recovery_candidate_pool
from src.domain.models.trade import TradeDirection
from tests.market_symbols import ALT_SYMBOL, ANCHOR, PAIR


def test_candidate_score_penalizes_same_symbol_in_recovery():
    metrics = {"trade_score": 0.55, "val_accuracy": 0.5, "raw_prob": 0.42}
    same = candidate_execution_score(metrics, recovery_active=True, symbol="SYM", last_loss_symbol="SYM")
    other = candidate_execution_score(metrics, recovery_active=True, symbol="SYM", last_loss_symbol=ALT_SYMBOL)
    assert same < other


def test_candidate_score_bonus_for_different_symbol():
    metrics = {"trade_score": 0.55, "execute": True, "raw_prob": 0.42}
    base = candidate_execution_score(metrics, recovery_active=True, symbol=ANCHOR)
    diversified = candidate_execution_score(metrics, recovery_active=True, symbol=ANCHOR, last_loss_symbol=ALT_SYMBOL)
    assert diversified >= base + 0.05


def test_recovery_score_ignores_direction_bias():
    call_metrics = {"raw_prob": 0.62, "execute": True}
    put_metrics = {"raw_prob": 0.38, "execute": True}
    call_score = candidate_execution_score(call_metrics, recovery_active=True, symbol="SYM")
    put_score = candidate_execution_score(put_metrics, recovery_active=True, symbol="SYM")
    assert abs(call_score - put_score) < 0.05


def test_select_best_prefers_higher_score_not_opposite_side():
    candidates = [
        (PAIR, TradeDirection.PUT, {"trade_score": 0.70, "val_accuracy": 0.55, "execute": False, "raw_prob": 0.40}),
        (ANCHOR, TradeDirection.CALL, {"trade_score": 0.72, "val_accuracy": 0.58, "execute": True, "raw_prob": 0.62}),
        (ANCHOR, TradeDirection.PUT, {"trade_score": 0.68, "val_accuracy": 0.56, "execute": True, "raw_prob": 0.38}),
    ]
    best = select_best_execution_candidate(candidates, last_loss_symbol=PAIR, recovery_active=True)
    assert best is not None
    assert best[0] == ANCHOR
    assert best[1] == TradeDirection.CALL


def test_recovery_candidate_pool_skips_blocked_symbol():
    candidates = [
        ("R_10", TradeDirection.CALL, {}),
        ("R_75", TradeDirection.PUT, {}),
    ]
    pool = recovery_candidate_pool(
        candidates, last_loss_symbol="R_10", recovery_active=True, skip_symbols=frozenset({"R_10"})
    )
    assert [c[0] for c in pool] == ["R_75"]


def test_select_mandatory_execution_candidate():
    candidates = [
        (ANCHOR, TradeDirection.CALL, {"trade_score": 0.8, "raw_prob": 0.7, "execute": True, "val_accuracy": 0.6}),
    ]
    orch = SimpleNamespace()
    best = select_mandatory_execution_candidate(orch, candidates, last_loss_symbol=None, recovery_active=False)
    assert best is not None
    assert best[1] == TradeDirection.CALL
