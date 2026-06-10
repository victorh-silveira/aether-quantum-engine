from unittest.mock import MagicMock, patch

import numpy as np

from src.application.services.deep_learning.dl_symbol_runtime import run_symbol_training
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
        "label_smoothing": 0.02,
        "label_min_move_pct": 0.00015,
        "early_stopping_patience": 3,
        "focal_gamma": 0.0,
        "calib_ratio": 0.15,
        "brier_untrained_floor": 0.99,
    }


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
        early_stopping_patience=5,
        progress_cb=lambda epoch, total, loss, acc: seen.append((epoch, total)),
    )
    assert result is not None
    assert seen[0] == (1, 3)
    assert all(total == 3 for _epoch, total in seen)


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
        "src.application.services.deep_learning.dl_symbol_runtime.train_model_walkforward",
        return_value=None,
    ) as mock_train:
        run_symbol_training(
            "R_50",
            runtime,
            prices,
            dl_config,
            params,
            100,
            orch,
            pair_prices=None,
            granularity=300,
        )
    progress_cb = mock_train.call_args.kwargs["progress_cb"]
    progress_cb(1, 2, 0.5123, 0.55)
    runtime["val_brier"] = 0.20
    with patch(
        "src.application.services.deep_learning.dl_symbol_runtime.train_model_walkforward",
        return_value=None,
    ) as mock_train_trained:
        run_symbol_training(
            "R_50",
            runtime,
            prices,
            dl_config,
            params,
            100,
            orch,
            pair_prices=None,
            granularity=300,
        )
    assert "progress_cb" in mock_train_trained.call_args.kwargs
