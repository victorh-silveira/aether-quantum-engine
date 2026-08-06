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


def test_train_walkforward_warns_when_oos_sharpness_collapses(caplog):
    import logging

    prices = np.sin(np.linspace(0, 12, 130)) + 10.0
    model = create_direction_model(arch="tcn")
    calls = {"n": 0}

    def fake_sharp(_probs):
        calls["n"] += 1
        return 0.05 if calls["n"] == 1 else 0.001

    with (
        patch(
            "src.application.services.deep_learning.dl_training.mean_sharpness",
            side_effect=fake_sharp,
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
    assert any("sharpness val_cal" in r.message for r in caplog.records)


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
            "OTC_SPC",
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
            "OTC_SPC",
            runtime,
            prices,
            dl_config,
            params,
            100,
            orch,
            granularity=900,
        )
    assert "progress_cb" in mock_train_trained.call_args.kwargs


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
            "OTC_SPC",
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
            "OTC_SPC",
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
            "OTC_SPC",
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
            "OTC_SPC",
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
            "OTC_SPC",
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
