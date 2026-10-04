import logging
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from torch import nn

from src.application.services.deep_learning.dl_feature_build import FEATURE_DIM
from src.application.services.deep_learning.dl_splits import purged_temporal_splits
from src.application.services.deep_learning.dl_symbol_train import run_symbol_training
from src.application.services.deep_learning.dl_training import fit_training_epochs, train_model_walkforward
from src.application.services.deep_learning.dl_training_loss import calculate_masked_loss
from src.application.services.deep_learning.model import create_direction_model


def _training_params():
    return {
        "lookback": 96,
        "validation_bars": 72,
        "epochs": 2,
        "arch": "tcn",
        "lr": 0.001,
        "weight_decay": 0.0001,
        "calib_ratio": 0.15,
        "label_horizon_bars": 1,
        "training_batch_size": 128,
        "training_log_every_n_epochs": 1,
    }


def test_train_walkforward_mini_batches(caplog):
    prices = np.sin(np.linspace(0, 12, 130)) + 10.0
    model = create_direction_model(arch="tcn")
    with caplog.at_level(logging.INFO, logger="AETH"):
        result = train_model_walkforward(
            model,
            prices,
            lookback=18,
            epochs=1,
            lr=0.001,
            validation_bars=14,
            batch_size=4,
        )
    assert result is not None
    assert "DL TREINO DIAG | train_n=" in caplog.text
    assert "majority_val=" in caplog.text


def test_train_walkforward_aligns_sparse_deltas_with_purged_train_split():
    sample_count = 120
    x_all = np.random.default_rng(42).normal(size=(sample_count, 8, FEATURE_DIM)).astype(np.float32)
    y_all = (np.arange(sample_count) % 2).astype(np.float32)
    masks = np.ones(sample_count, dtype=np.float32)
    deltas = np.linspace(-0.02, 0.02, sample_count, dtype=np.float32)
    train_sl, _, _ = purged_temporal_splits(sample_count, 20, embargo=1)
    model = create_direction_model(arch="tcn")
    with (
        patch(
            "src.application.services.deep_learning.dl_training.extract_sequences_and_deltas",
            return_value=(x_all, y_all, masks, deltas),
        ) as extract_mock,
        patch(
            "src.application.services.deep_learning.dl_training.run_preflight_linear_baseline",
            return_value=(0.5, 0.5),
        ),
        patch(
            "src.application.services.deep_learning.dl_training.fit_training_epochs",
            wraps=fit_training_epochs,
        ) as fit_mock,
    ):
        result = train_model_walkforward(
            model,
            np.linspace(100.0, 101.0, 140),
            lookback=8,
            epochs=1,
            lr=0.001,
            validation_bars=20,
            label_mode="triple_barrier",
            asymmetric_payout_loss=False,
        )
    assert result is not None
    assert extract_mock.call_args.kwargs["filter_active"] is True
    np.testing.assert_array_equal(fit_mock.call_args.kwargs["delta_train"], deltas[train_sl])
    assert fit_mock.call_args.kwargs["asymmetric_payout_loss"] is False


def test_train_walkforward_preflight_floor_blocks_weak_linear_signal(caplog):
    count = 120
    x_all = np.zeros((count, 8, FEATURE_DIM), dtype=np.float32)
    y_all = (np.arange(count) % 2).astype(np.float32)
    masks = np.ones(count, dtype=np.float32)
    deltas = np.zeros(count, dtype=np.float32)
    model = create_direction_model(arch="tcn")
    with (
        patch(
            "src.application.services.deep_learning.dl_training.extract_sequences_and_deltas",
            return_value=(x_all, y_all, masks, deltas),
        ),
        patch(
            "src.application.services.deep_learning.dl_training.run_preflight_linear_baseline",
            return_value=(0.5, 0.4),
        ),
        caplog.at_level(logging.WARNING, logger="AETH"),
    ):
        result = train_model_walkforward(
            model,
            np.linspace(100.0, 101.0, 140),
            lookback=8,
            epochs=1,
            lr=0.001,
            validation_bars=20,
            dl_config={"training_quality": {"min_linear_preflight_acc": 0.5}},
        )
    assert result is None
    assert "abortando treino TCN" in caplog.text


def test_loss_comparison_override_reaches_train_and_validation():
    rng = np.random.default_rng(7)
    model = nn.Linear(4, 1)
    x_train = rng.normal(size=(16, 4)).astype(np.float32)
    x_val = rng.normal(size=(8, 4)).astype(np.float32)
    y_train = np.array([0.0, 1.0] * 8, dtype=np.float32)
    y_val = np.array([0.0, 1.0] * 4, dtype=np.float32)
    with patch(
        "src.application.services.deep_learning.dl_training_epochs._masked_loss",
        wraps=calculate_masked_loss,
    ) as loss_mock:
        fit_training_epochs(
            model,
            x_train,
            y_train,
            np.ones(16, dtype=np.float32),
            [1.0] * 16,
            x_val,
            y_val,
            np.ones(8, dtype=np.float32),
            torch.device("cpu"),
            epochs=1,
            batch_size=16,
            lr=0.001,
            weight_decay=0.0,
            label_smoothing=0.0,
            focal_gamma=0.0,
            asymmetric_payout_loss=False,
        )
    assert loss_mock.call_count == 2
    assert all(call.kwargs["asymmetric_payout_loss"] is False for call in loss_mock.call_args_list)


def test_train_walkforward_falls_back_identity_when_oos_sharpness_collapses(caplog):
    import logging

    from src.application.services.deep_learning.dl_calibration import CalibratorState

    prices = np.sin(np.linspace(0, 12, 130)) + 10.0
    model = create_direction_model(arch="tcn")
    collapsing = CalibratorState(method="isotonic", isotonic_x=(0.2, 0.8), isotonic_y=(0.495, 0.505))

    with (
        patch(
            "src.application.services.deep_learning.dl_training.fit_calibrator",
            return_value=collapsing,
        ),
        caplog.at_level(logging.WARNING),
    ):
        result = train_model_walkforward(
            model,
            prices,
            lookback=18,
            epochs=1,
            lr=0.001,
            validation_bars=14,
            dl_config={"calibration": {"min_oos_sharpness": 0.01, "min_calibration_sharpness": 0.01}},
        )
    assert result is not None
    assert result.calibrator.method == "identity"
    assert result.oos_sharpness >= 0.01
    assert "usando identity" in caplog.text or any("usando identity" in r.message for r in caplog.records)


def test_train_walkforward_reports_progress():
    prices = np.sin(np.linspace(0, 12, 130)) + 10.0
    model = create_direction_model(arch="tcn")
    seen = []
    result = train_model_walkforward(
        model,
        prices,
        lookback=18,
        epochs=3,
        lr=0.001,
        validation_bars=14,
        progress_cb=lambda epoch, total, loss, acc: seen.append((epoch, total)),
    )
    assert result is not None
    assert seen[0] == (1, 3)
    assert all(total == 3 for _epoch, total in seen)


def test_train_walkforward_oos_sharpness_without_validation_split():
    prices = np.sin(np.linspace(0, 12, 130)) + 10.0
    model = create_direction_model(arch="tcn")
    empty_val = np.asarray([], dtype=np.float32)

    with patch(
        "src.application.services.deep_learning.dl_training._model_raw_prob",
        side_effect=[empty_val, empty_val],
    ):
        result = train_model_walkforward(
            model,
            prices,
            lookback=18,
            epochs=1,
            lr=0.001,
            validation_bars=14,
        )
    assert result is not None
    assert result.oos_sharpness >= 0.0


def test_run_symbol_training_forwards_progress_callback():
    orch = MagicMock()
    runtime = {
        "model": create_direction_model(arch="tcn"),
        "norm_stats": MagicMock(),
        "calibrator": MagicMock(),
    }
    dl_config = {"deploy_gate": {"enabled": False}}
    params = _training_params()
    prices = np.linspace(1.0, 2.0, 80)
    with patch(
        "src.application.services.deep_learning.dl_symbol_train.train_model_walkforward",
        return_value=None,
    ) as mock_train:
        run_symbol_training(
            "R_10",
            runtime,
            prices,
            dl_config,
            params,
            100,
            orch,
            granularity=900,
        )
    progress_cb = mock_train.call_args.kwargs["progress_cb"]
    progress_cb(1, 2, 0.5123, 0.55)
    runtime["val_brier"] = 0.20
    with patch(
        "src.application.services.deep_learning.dl_symbol_train.train_model_walkforward",
        return_value=None,
    ) as mock_train_trained:
        run_symbol_training(
            "R_10",
            runtime,
            prices,
            dl_config,
            params,
            100,
            orch,
            granularity=900,
        )
    assert "progress_cb" in mock_train_trained.call_args.kwargs
