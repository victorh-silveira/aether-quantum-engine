from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.application.services.orchestrator.execution_collect_helpers import (
    _finalize_force_trade_candidate,
    log_execution_decision,
    mandatory_fallback_if_empty,
)
from src.domain.models.market_data import Candle
from src.domain.models.trade import TradeDirection
from tests.market_symbols import ANCHOR


def test_helper_mandatory_fallback_if_empty_returns_existing_candidates():
    exec_mgr = SimpleNamespace(
        _trade_symbols=lambda: [ANCHOR],
        orch=SimpleNamespace(risk_manager=SimpleNamespace(consecutive_losses_linear=0)),
    )
    existing = [(ANCHOR, TradeDirection.CALL, {"trade_score": 0.6})]
    kept = mandatory_fallback_if_empty(
        exec_mgr,
        {},
        existing,
        mandatory=True,
        recovery_active=False,
        last_loss=None,
        skip_symbols=frozenset(),
        min_signal=0.5,
        min_val=0.5,
    )
    assert kept is existing


def test_finalize_force_trade_candidate_none_and_missing_entry():
    exec_mgr = SimpleNamespace(orch=SimpleNamespace(config={}, _active_cycle_id=1, risk_manager=None))
    assert _finalize_force_trade_candidate(exec_mgr, {}, None) is None
    forced = (ANCHOR, TradeDirection.CALL, {"execute": True})
    assert _finalize_force_trade_candidate(exec_mgr, {ANCHOR: "bad"}, forced) is forced
    assert _finalize_force_trade_candidate(exec_mgr, None, forced) is forced


def test_log_execution_decision_with_live_candle_and_movement():
    candle = Candle(
        symbol=ANCHOR,
        open=100.0,
        high=105.0,
        low=98.0,
        close=103.0,
        time=datetime.now(),
        epoch=1000,
    )
    stream = SimpleNamespace(micro_candles={ANCHOR: [candle]}, tick_buffer=None)
    orch = SimpleNamespace(stream=stream)
    logger = MagicMock()
    exec_mgr = SimpleNamespace(orch=orch, logger=logger)

    metrics = {
        "predicted_movement_delta": 0.0025,
        "predicted_movement_side": "CALL",
        "movement_confluence": True,
        "movement_atr_ratio": 1.5,
        "trade_score": 0.65,
        "calibrated_prob": 0.65,
    }
    best = (ANCHOR, TradeDirection.CALL, metrics)
    log_execution_decision(exec_mgr, "C0001", best, [best], 0.65)

    assert logger.info.call_count >= 3
    logged_lines = [call.args[1] if len(call.args) > 1 else call.args[0] for call in logger.info.call_args_list]
    joined = " ".join(str(entry_line) for entry_line in logged_lines)
    assert "[LIVE_CANDLE]" in joined
    assert "[NEXT_MOVE]" in joined
    assert "[MARKET]" in joined
    assert "[DECISION]" in joined
