from unittest.mock import MagicMock, patch

import numpy as np

from src.application.services.deep_learning.dl_symbol_train import run_symbol_training
from src.application.services.deep_learning.dl_training import train_model_walkforward
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


def test_train_walkforward_mini_batches():
    prices = np.sin(np.linspace(0, 12, 130)) + 10.0
    model = create_direction_model(arch="tcn")
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
        caplog.at_level(logging.WARNING, logger="AETH"),
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
    assert any("usando identity" in r.message for r in caplog.records)


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
