import logging
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.application.services.deep_learning.decision_bridge import (
    _apply_deploy_gate,
    _collect_symbol_decision,
    _insufficient_data_entry,
    _log_retrain_batch,
    _min_dl_history_len,
)
from src.application.services.deep_learning.dl_bridge_helpers import build_decision_entry, resample_m1_to_m15
from src.domain.models.trade import TradeDirection
from tests.unit.application.dl_collect_fixtures import MockOrchestrator


def test_min_dl_history_len_standard():
    assert _min_dl_history_len({"lookback": 32, "validation_bars": 10, "training_history_bars": 100}) == 100


def test_insufficient_data_entry_gate_reason():
    entry = _insufficient_data_entry()
    assert entry["metrics"]["gate_reason"] == "data"
    assert entry["direction"] is None


def test_build_decision_entry_includes_train_loss():
    entry = build_decision_entry(
        TradeDirection.CALL,
        0.6,
        execute=True,
        val_accuracy=0.55,
        edge=0.08,
        train_loss=0.1234,
        contract_duration=5,
    )
    assert "loss=0.1234" in entry["metrics"]["llm_note"]
    assert entry["metrics"]["duration"] == 5


def test_apply_deploy_gate_blocks_when_not_ok():
    entry = {"metrics": {"execute": True}}
    runtime = {"deploy_ok": False}
    out = _apply_deploy_gate(entry, runtime, {"deploy_gate": {"enabled": True}})
    assert out["metrics"]["execute"] is False
    assert out["metrics"]["gate_reason"] == "deploy"
    assert out["metrics"]["deploy_ok"] is False


def test_apply_deploy_gate_force_ok_allows_execution():
    entry = {"metrics": {"execute": True}}
    runtime = {"deploy_ok": False}
    out = _apply_deploy_gate(entry, runtime, {"deploy_gate": {"enabled": True, "force_ok": True}})
    assert out["metrics"]["execute"] is True
    assert out["metrics"]["deploy_ok"] is True
    assert out["metrics"].get("gate_reason") != "deploy"


def test_log_retrain_batch_empty_and_nonempty(caplog):
    _log_retrain_batch([], "bootstrap", {"training_history_bars": 5})
    with caplog.at_level(logging.DEBUG):
        _log_retrain_batch(["R_10", "R_50"], "new_candle", {"training_history_bars": 5})
    assert "retreino" in caplog.text


@pytest.mark.asyncio
async def test_collect_symbol_decision_insufficient_history():
    orch = MagicMock()
    orch.stream.get_numpy_series = MagicMock(return_value=np.array([1.0, 2.0]))
    entry, reason = await _collect_symbol_decision(
        orch,
        "R_10",
        dl_config={},
        params={"training_history_bars": 32},
        min_len=10,
        granularity=60,
    )
    assert entry["metrics"]["gate_reason"] == "data"
    assert reason is None


@pytest.mark.asyncio
async def test_collect_symbol_decision_insufficient_after_slice():
    prices = np.linspace(1.0, 2.0, 120)
    orch = MagicMock()
    orch.stream.get_numpy_series = MagicMock(return_value=prices)
    with patch(
        "src.application.services.deep_learning.decision_bridge.slice_dl_ohlc_window",
        return_value=(prices[:5], None, None, None),
    ):
        entry, reason = await _collect_symbol_decision(
            orch,
            "R_10",
            dl_config={},
            params={"training_history_bars": 32, "lookback": 32, "validation_bars": 10},
            min_len=100,
            granularity=60,
        )
    assert entry["metrics"]["gate_reason"] == "data"
    assert reason is None


@pytest.mark.asyncio
async def test_collect_symbol_decision_full_path():
    prices = np.sin(np.linspace(0, 10, 90)) + 10.0
    orch = MockOrchestrator(["R_10"], prices, train_mode=True)
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {"execute": True, "conviction": 0.62, "trade_score": 0.62, "val_accuracy": 0.52},
    }
    runtime = {
        "model": MagicMock(),
        "norm_stats": MagicMock(),
        "val_accuracy": 0.52,
        "val_brier": 0.25,
        "calibrator": None,
        "lookback": 32,
        "deploy_ok": True,
        "deploy_win_rate": 0.5,
        "last_candle_epoch": 0,
    }
    with (
        patch(
            "src.application.services.deep_learning.decision_bridge.should_retrain_symbol",
            return_value=(True, "bootstrap"),
        ),
        patch(
            "src.application.services.deep_learning.decision_bridge.enqueue_deferred_symbol_training"
        ) as mock_enqueue,
        patch(
            "src.application.services.deep_learning.decision_bridge.predict_symbol_decision",
            new_callable=AsyncMock,
            return_value=entry,
        ),
        patch(
            "src.application.services.deep_learning.decision_bridge.get_symbol_runtime",
            return_value=runtime,
        ),
        patch(
            "src.application.services.deep_learning.decision_bridge.candle_epoch",
            return_value=1000,
        ),
    ):
        out, reason = await _collect_symbol_decision(
            orch,
            "R_10",
            dl_config={"deploy_gate": {"enabled": False}},
            params={"training_history_bars": 60, "lookback": 32},
            min_len=30,
            granularity=60,
        )
    assert reason == "bootstrap"
    assert out["direction"] == TradeDirection.CALL
    mock_enqueue.assert_called_once()


def test_resample_m1_to_m15():
    prices = np.array([1.0, 2.0])
    res_p, res_o, res_h, res_l = resample_m1_to_m15(prices, None, None, None)
    assert np.array_equal(res_p, prices)

    prices = np.linspace(1.0, 30.0, 30)
    open_val = np.linspace(1.0, 30.0, 30)
    high_val = np.linspace(1.5, 30.5, 30)
    low_val = np.linspace(0.5, 29.5, 30)
    res_p, res_o, res_h, res_l = resample_m1_to_m15(prices, open_val, high_val, low_val)
    assert len(res_p) == 2
    assert len(res_o) == 2
    assert len(res_h) == 2
    assert len(res_l) == 2
    assert res_h[0] == np.max(high_val[0:15])
    assert res_h[1] == np.max(high_val[15:30])

    res_p, res_o, res_h, res_l = resample_m1_to_m15(prices, open_val, None, None)
    assert res_h is None
    assert res_l is None


@pytest.mark.asyncio
async def test_collect_symbol_decision_uses_macro_buffer():
    prices = np.linspace(1.0, 10.0, 960)
    orch = MockOrchestrator(["R_10"], prices, train_mode=True)
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {"execute": True, "conviction": 0.62, "trade_score": 0.62, "val_accuracy": 0.52},
    }
    runtime = {
        "model": MagicMock(),
        "norm_stats": MagicMock(),
        "val_accuracy": 0.52,
        "val_brier": 0.25,
        "calibrator": None,
        "lookback": 32,
        "deploy_ok": True,
        "deploy_win_rate": 0.5,
        "last_candle_epoch": 0,
        "trained_granularity": 900,
    }
    with (
        patch(
            "src.application.services.deep_learning.decision_bridge.should_retrain_symbol",
            return_value=(False, ""),
        ),
        patch(
            "src.application.services.deep_learning.decision_bridge.predict_symbol_decision",
            new_callable=AsyncMock,
            return_value=entry,
        ),
        patch(
            "src.application.services.deep_learning.decision_bridge.get_symbol_runtime",
            return_value=runtime,
        ),
        patch(
            "src.application.services.deep_learning.decision_bridge.candle_epoch",
            return_value=1000,
        ),
    ):
        out, reason = await _collect_symbol_decision(
            orch,
            "R_10",
            dl_config={"deploy_gate": {"enabled": False}},
            params={"training_history_bars": 60, "lookback": 32},
            min_len=30,
            granularity=900,
        )
    assert out["direction"] == TradeDirection.CALL
