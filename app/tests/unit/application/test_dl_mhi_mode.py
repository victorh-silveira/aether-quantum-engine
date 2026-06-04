from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.application.services.deep_learning.decision_bridge import collect_deep_learning_decisions
from src.application.services.deep_learning.dl_features import _extract_single_mhi_window, extract_sequences
from src.application.services.deep_learning.dl_params import parse_dl_params
from src.application.services.deep_learning.dl_splits import (
    _purged_temporal_splits_mhi,
    purged_temporal_splits,
)


def test_parse_dl_params_mhi_mode():
    params = parse_dl_params(
        {"mhi_mode": True, "lookback": 5, "training_history_bars": 5, "validation_bars": 1},
        {"granularity": 60},
    )
    assert params["mhi_mode"] is True
    assert params["lookback"] == 5
    assert params["training_history_bars"] == 5
    assert params["validation_bars"] == 1
    assert params["compact_mhi"] is True


def test_extract_single_mhi_window_with_ohlc():
    prices = np.linspace(100.0, 104.0, 5, dtype=np.float64)
    open_ = prices - 0.1
    high = prices + 0.2
    low = prices - 0.2
    x, y, mask = _extract_single_mhi_window(
        prices,
        5,
        label_min_move_pct=0.0,
        granularity=60,
        pair_prices=None,
        require_pair_label=False,
        sym_is_bull=True,
        open_=open_,
        high=high,
        low=low,
    )
    assert x.shape == (1, 5, x.shape[2])
    assert y[0] == 1.0
    assert mask[0] == 1.0


def test_extract_sequences_compact_mhi_five_bars():
    prices = np.linspace(100.0, 104.0, 5, dtype=np.float64)
    x, y, mask = extract_sequences(prices, 5, compact_mhi=True, granularity=60)
    assert x.shape == (1, 5, x.shape[2])
    assert len(y) == 1


def test_purged_splits_compact_mhi():
    assert purged_temporal_splits(0, 1, compact_mhi=True) is None
    assert purged_temporal_splits(1, 1, compact_mhi=True) is not None
    splits = purged_temporal_splits(2, 1, compact_mhi=True)
    assert splits is not None
    train_sl, val_sl, calib_sl = splits
    assert train_sl.stop <= val_sl.start
    assert purged_temporal_splits(8, 1, compact_mhi=True) is not None


def test_extract_sequences_compact_mhi_with_pair_label():
    prices = np.linspace(100.0, 104.0, 5, dtype=np.float64)
    peer = prices * 1.001
    x, y, mask = extract_sequences(
        prices,
        5,
        compact_mhi=True,
        granularity=60,
        pair_prices=peer,
        require_pair_label=True,
        sym_is_bull=True,
    )
    assert x.shape[0] == 1
    assert mask[0] in (0.0, 1.0)


def test_purged_temporal_splits_mhi_direct_large_sample():
    splits = _purged_temporal_splits_mhi(12, 2)
    assert splits is not None
    train_sl, val_sl, calib_sl = splits
    assert train_sl.stop > 0
    assert val_sl.stop > val_sl.start


def test_purged_splits_compact_mhi_small_holdout_branch():
    splits = purged_temporal_splits(4, 1, compact_mhi=True)
    assert splits is not None
    train_sl, val_sl, calib_sl = splits
    assert train_sl.stop >= 1


@pytest.mark.asyncio
async def test_collect_decisions_mhi_with_five_bars():
    orch = MagicMock()
    orch.symbols = ["R_50"]
    orch.config = {
        "deep_learning": {
            "enabled": True,
            "mhi_mode": True,
            "lookback": 5,
            "training_history_bars": 5,
            "validation_bars": 1,
            "deploy_gate": {"enabled": False},
            "train_on_new_candle_only": False,
        },
        "data_handler": {"granularity": 60},
        "risk_management": {"params": {"duration": 1, "duration_unit": "m"}},
    }
    orch.risk_manager.pending_loss = {}
    prices = np.linspace(100.0, 104.0, 5, dtype=np.float64)
    orch.stream.get_numpy_series = MagicMock(return_value=prices.copy())
    orch.stream.get_last_candle_epoch = MagicMock(return_value=1000)
    orch.stream.candles = {"R_50": []}
    with (
        patch(
            "src.application.services.deep_learning.decision_bridge.run_symbol_training",
            return_value=(None, None),
        ),
        patch(
            "src.application.services.deep_learning.decision_bridge.predict_symbol_decision",
            return_value={"direction": None, "metrics": {"execute": False, "conviction": 0.5}},
        ),
    ):
        decisions = await collect_deep_learning_decisions(orch)
    assert "R_50" in decisions


@pytest.mark.asyncio
async def test_collect_decisions_non_mhi_insufficient_history():
    orch = MagicMock()
    orch.symbols = ["R_50"]
    orch.config = {
        "deep_learning": {
            "enabled": True,
            "lookback": 32,
            "training_history_bars": 100,
            "validation_bars": 10,
        },
        "data_handler": {"granularity": 60},
        "risk_management": {"params": {"duration": 1, "duration_unit": "m"}},
    }
    orch.risk_manager.pending_loss = {}
    orch.stream.get_numpy_series = MagicMock(return_value=np.arange(10.0, 20.0))
    orch.stream.candles = {}
    decisions = await collect_deep_learning_decisions(orch)
    assert decisions["R_50"]["metrics"]["gate_reason"] == "data"


def test_purged_splits_mhi_invalid_slice_returns_none():
    with patch(
        "src.application.services.deep_learning.dl_splits.splits_valid",
        return_value=False,
    ):
        assert _purged_temporal_splits_mhi(6, 1) is None
