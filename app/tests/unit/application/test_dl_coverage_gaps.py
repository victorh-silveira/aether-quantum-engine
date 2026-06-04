from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.application.services.auth_manager import AuthManager
from src.application.services.deep_learning.dl_bridge_helpers import parse_dl_params
from src.application.services.deep_learning.dl_calibration import (
    brier_score,
    calibrate_conviction,
    calibrator_from_dict,
    expected_calibration_error,
    fit_platt,
)
from src.application.services.deep_learning.dl_outcomes import record_symbol_outcome, sample_weights_for_symbol
from src.application.services.deep_learning.dl_sim_backtest import direction_wins, run_dl_walkforward
from src.application.services.deep_learning.dl_splits import purged_temporal_splits, splits_valid
from src.application.services.deep_learning.dl_symbol_runtime import run_symbol_training
from src.application.services.deep_learning.dl_tcn import TemporalDirectionClassifier, _Chomp1d
from src.application.services.deep_learning.dl_training import train_model_walkforward
from src.application.services.deep_learning.model import (
    INPUT_DIM,
    FeatureNormStats,
    MarketDirectionClassifier,
    create_direction_model,
    load_model_checkpoint,
    model_accuracy,
    normalize_features,
    normalize_sequences,
)
from src.domain.models.trade import TradeDirection


def test_auth_manager_get_pat(monkeypatch):
    monkeypatch.setenv("AETHER_DERIV_PAT", "pat_live_test")
    auth = AuthManager(mode="live")
    assert auth.get_pat() == "pat_live_test"


def test_calibrate_conviction_legacy():
    assert calibrate_conviction(0.8, 0.6, 1.0) > 0.5
    assert calibrator_from_dict(None).temperature == 1.0
    assert expected_calibration_error([], []) == 1.0
    assert brier_score([], []) == 1.0
    assert fit_platt([], []) == (1.0, 0.0)


def test_outcome_weights_dampen_after_win_streak():
    orch = type("O", (), {})()
    for _ in range(6):
        record_symbol_outcome(orch, "RDBULL", won=True)
    weights = sample_weights_for_symbol(orch, "RDBULL", 12)
    assert min(weights[-4:]) < 1.0


def test_purged_splits_edge_cases():
    assert purged_temporal_splits(15, 5) is None
    splits = purged_temporal_splits(500, 30, calib_ratio=0.1)
    assert splits is not None


def test_purged_splits_invalid_holdout_returns_none():
    assert purged_temporal_splits(26, 10) is not None
    assert purged_temporal_splits(30, 28, calib_ratio=0.5) is None
    assert splits_valid(10, 12, 20, 20) is False


def test_purged_splits_rejects_invalid_slice_ranges(monkeypatch):
    monkeypatch.setattr(
        "src.application.services.deep_learning.dl_splits.splits_valid",
        lambda *_args: False,
    )
    assert purged_temporal_splits(120, 20) is None


def test_dl_sim_backtest_edge_paths():
    prices = np.array([10.0, 11.0])
    assert not direction_wins(TradeDirection.CALL, prices, 1)
    params = parse_dl_params({"lookback": 20, "validation_bars": 15})
    with pytest.raises(RuntimeError):
        run_dl_walkforward(prices, params)


def test_tcn_chomp_and_2d_input():
    chomp = _Chomp1d(0)
    x = torch.randn(2, 8, 10)
    assert chomp(x).shape == x.shape
    chomp_trim = _Chomp1d(2)
    assert chomp_trim(x).shape[2] == x.shape[2] - 2
    model = TemporalDirectionClassifier(input_dim=INPUT_DIM)
    out = model(torch.randn(3, INPUT_DIM))
    assert out.shape == (3, 1)
    out3d = model(torch.randn(2, 12, INPUT_DIM))
    assert out3d.shape == (2, 1)


def test_train_with_focal_loss():
    prices = np.sin(np.linspace(0, 12, 130)) + 10.0
    model = create_direction_model(arch="tcn")
    result = train_model_walkforward(
        model,
        prices,
        lookback=18,
        epochs=2,
        lr=0.001,
        validation_bars=14,
        focal_gamma=2.0,
    )
    assert result is not None


def test_model_accuracy_mask_and_normalize_2d():
    model = create_direction_model(arch="legacy")
    assert isinstance(model, MarketDirectionClassifier)
    x = np.random.randn(4, 12, INPUT_DIM).astype(np.float32)
    y = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    assert model_accuracy(model, x, y) >= 0.0
    assert model_accuracy(model, x, y, np.zeros(4, dtype=np.float32)) == 0.0
    mask_sparse = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    acc_sparse = model_accuracy(model, x, y, mask_sparse)
    assert 0.0 <= acc_sparse <= 1.0
    flat = np.random.randn(3, INPUT_DIM).astype(np.float32)
    stats = FeatureNormStats(mean=np.zeros(INPUT_DIM, dtype=np.float32), std=np.ones(INPUT_DIM, dtype=np.float32))
    assert normalize_features(flat, stats).shape == flat.shape
    assert normalize_sequences(flat, stats).shape == flat.shape


def test_load_corrupted_checkpoint(tmp_path):
    bad = tmp_path / "corrupt.pth"
    bad.write_bytes(b"not-a-checkpoint")
    assert load_model_checkpoint(bad) is None


def test_run_symbol_training_when_walkforward_unavailable():
    orch = MagicMock()
    runtime = {
        "model": create_direction_model(arch="tcn"),
        "norm_stats": MagicMock(),
        "calibrator": MagicMock(),
    }
    dl_config = {"deploy_gate": {"enabled": False}}
    params = {
        "lookback": 96,
        "validation_bars": 72,
        "epochs": 2,
        "arch": "tcn",
        "lr": 0.001,
        "weight_decay": 0.0001,
        "label_smoothing": 0.02,
        "label_min_move_pct": 0.00015,
        "early_stopping_patience": 3,
        "focal_gamma": 0.0,
        "calib_ratio": 0.15,
    }
    prices = np.linspace(1.0, 2.0, 80)
    with patch(
        "src.application.services.deep_learning.dl_symbol_runtime.train_model_walkforward",
        return_value=None,
    ):
        stats, loss = run_symbol_training(
            "RDBULL",
            runtime,
            prices,
            dl_config,
            params,
            100,
            orch,
            pair_prices=None,
            granularity=300,
        )
    assert stats is runtime["norm_stats"]
    assert loss is None
    assert runtime["deploy_ok"] is False
    assert runtime["val_accuracy"] == 0.0


def test_run_symbol_training_handles_training_exception():
    orch = MagicMock()
    runtime = {
        "model": create_direction_model(arch="tcn"),
        "norm_stats": MagicMock(),
        "calibrator": MagicMock(),
    }
    dl_config = {"deploy_gate": {"enabled": False}}
    params = {
        "lookback": 96,
        "validation_bars": 72,
        "epochs": 2,
        "arch": "tcn",
        "lr": 0.001,
        "weight_decay": 0.0001,
        "label_smoothing": 0.02,
        "label_min_move_pct": 0.00015,
        "early_stopping_patience": 3,
        "focal_gamma": 0.0,
        "calib_ratio": 0.15,
    }
    prices = np.linspace(1.0, 2.0, 80)
    with patch(
        "src.application.services.deep_learning.dl_symbol_runtime.train_model_walkforward",
        side_effect=RuntimeError("fail"),
    ):
        stats, loss = run_symbol_training(
            "RDBULL",
            runtime,
            prices,
            dl_config,
            params,
            100,
            orch,
            pair_prices=None,
            granularity=300,
        )
    assert stats is runtime["norm_stats"]
    assert loss is None
    assert runtime["deploy_ok"] is False
