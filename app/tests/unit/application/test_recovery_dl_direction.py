from src.application.services.execution_symbols import select_best_execution_candidate
from src.application.services.execution_symbols_recovery import recovery_candidate_pool
from src.domain.models.trade import TradeDirection
from tests.market_symbols import ANCHOR, HEDGE_PEER_SYMBOL, PAIR


def test_recovery_candidate_pool_keeps_all_directions():
    candidates = [
        (PAIR, TradeDirection.PUT, {"execute": True, "dl_direction": "PUT"}),
        (ANCHOR, TradeDirection.PUT, {"execute": True, "dl_direction": "PUT"}),
        (HEDGE_PEER_SYMBOL, TradeDirection.CALL, {"execute": True, "dl_direction": "CALL"}),
    ]
    result = recovery_candidate_pool(candidates, last_loss_symbol=PAIR, recovery_active=True)
    assert len(result) == 3


def test_recovery_select_prefers_different_symbol_after_put_loss():
    candidates = [
        ("R_10", TradeDirection.PUT, {"execute": True, "trade_score": 0.40, "raw_prob": 0.40, "val_accuracy": 0.55}),
        ("R_75", TradeDirection.PUT, {"execute": True, "trade_score": 0.65, "val_accuracy": 0.55, "raw_prob": 0.45}),
    ]
    best = select_best_execution_candidate(candidates, last_loss_symbol="R_10", recovery_active=True)
    assert best is not None
    assert best[0] == "R_75"


def test_select_best_picks_stronger_tcn_score_without_flip_bias():
    candidates = [
        ("R_10", TradeDirection.CALL, {"trade_score": 0.45, "val_accuracy": 0.55, "execute": True, "raw_prob": 0.58}),
        ("R_75", TradeDirection.PUT, {"trade_score": 0.64, "val_accuracy": 0.80, "execute": True, "raw_prob": 0.35}),
    ]
    best = select_best_execution_candidate(candidates, last_loss_symbol="R_10", recovery_active=True)
    assert best is not None
    assert best[0] == "R_75"
    assert best[1] == TradeDirection.PUT
