from unittest.mock import MagicMock, patch

import numpy as np

from src.application.services.deep_learning.dl_symbol_train import run_symbol_training
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


def test_run_symbol_training_skips_throttled_progress_logs():
    orch = MagicMock()
    runtime = {
        "model": create_direction_model(arch="tcn"),
        "norm_stats": MagicMock(),
        "calibrator": MagicMock(),
        "val_brier": 1.0,
    }
    dl_config = {"deploy_gate": {"enabled": False}}
    params = _training_params()
    params["training_log_every_n_epochs"] = 4
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
    with patch("src.application.services.deep_learning.dl_symbol_train.logger.log") as mock_log:
        progress_cb(3, 10, 0.5123, 0.55)
        mock_log.assert_not_called()
        progress_cb(4, 10, 0.5123, 0.55)
        mock_log.assert_called_once()


def test_run_symbol_training_when_walkforward_unavailable():
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
    ):
        stats, loss = run_symbol_training(
            "R_10",
            runtime,
            prices,
            dl_config,
            params,
            100,
            orch,
            granularity=900,
        )
    assert stats is runtime["norm_stats"]
    assert loss is None
    assert runtime["deploy_ok"] is False


def test_run_symbol_training_clears_cuda_after_cuda_failure():
    orch = MagicMock()
    runtime = {
        "model": create_direction_model(arch="tcn"),
        "norm_stats": MagicMock(),
        "calibrator": MagicMock(),
    }
    dl_config = {"deploy_gate": {"enabled": False}}
    params = _training_params()
    prices = np.linspace(1.0, 2.0, 80)
    with (
        patch(
            "src.application.services.deep_learning.dl_symbol_train.train_model_walkforward",
            side_effect=RuntimeError("CUDA error: device-side assert triggered"),
        ),
        patch("src.application.services.deep_learning.dl_symbol_train.torch.cuda.is_available", return_value=True),
        patch("src.application.services.deep_learning.dl_symbol_train.torch.cuda.synchronize") as mock_sync,
        patch("src.application.services.deep_learning.dl_symbol_train.torch.cuda.empty_cache") as mock_empty,
    ):
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
    mock_sync.assert_called_once()
    mock_empty.assert_called_once()
    assert runtime["deploy_ok"] is False


def test_run_symbol_training_cuda_cleanup_failure_is_ignored():
    orch = MagicMock()
    runtime = {
        "model": create_direction_model(arch="tcn"),
        "norm_stats": MagicMock(),
        "calibrator": MagicMock(),
    }
    dl_config = {"deploy_gate": {"enabled": False}}
    params = _training_params()
    prices = np.linspace(1.0, 2.0, 80)
    with (
        patch(
            "src.application.services.deep_learning.dl_symbol_train.train_model_walkforward",
            side_effect=RuntimeError("CUDA error: device-side assert triggered"),
        ),
        patch("src.application.services.deep_learning.dl_symbol_train.torch.cuda.is_available", return_value=True),
        patch(
            "src.application.services.deep_learning.dl_symbol_train.torch.cuda.synchronize",
            side_effect=RuntimeError("sync fail"),
        ),
    ):
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
    assert runtime["deploy_ok"] is False


def test_run_symbol_training_handles_training_exception():
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
        side_effect=RuntimeError("fail"),
    ):
        stats, loss = run_symbol_training(
            "R_10",
            runtime,
            prices,
            dl_config,
            params,
            100,
            orch,
            granularity=900,
        )
    assert stats is runtime["norm_stats"]
    assert loss is None
    assert runtime["deploy_ok"] is False


def test_run_symbol_training_logs_preview_label_balance():
    orch = MagicMock()
    runtime = {
        "model": create_direction_model(arch="tcn"),
        "norm_stats": MagicMock(),
        "calibrator": MagicMock(),
    }
    dl_config = {"deploy_gate": {"enabled": False}}
    params = _training_params()
    prices = np.linspace(1.0, 2.0, 80)
    y_preview = np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float64)
    with (
        patch(
            "src.application.services.deep_learning.dl_symbol_train.extract_sequences",
            return_value=(None, y_preview, None),
        ),
        patch(
            "src.application.services.deep_learning.dl_symbol_train.train_model_walkforward",
            return_value=None,
        ),
        patch("src.application.services.deep_learning.dl_symbol_train.logger.log") as mock_log,
    ):
        run_symbol_training(
            "R_10",
            runtime,
            prices,
            dl_config,
            params,
            100,
            orch,
            granularity=120,
        )
    assert any("labels up" in str(call.args[1]) for call in mock_log.call_args_list)
