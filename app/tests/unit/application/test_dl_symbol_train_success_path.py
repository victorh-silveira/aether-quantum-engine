from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from src.application.services.deep_learning.dl_symbol_train import run_symbol_training
from src.application.services.deep_learning.model import create_direction_model
from tests.unit.application.test_dl_training_progress import _training_params


def test_run_symbol_training_persists_successful_train(tmp_path):
    orch = MagicMock()
    orch.infra = MagicMock(enabled=False)
    runtime = {
        "model": create_direction_model(arch="tcn"),
        "norm_stats": MagicMock(),
        "calibrator": MagicMock(),
    }
    train_result = SimpleNamespace(
        norm_stats=MagicMock(),
        val_accuracy=0.61,
        calibrator=None,
        val_brier=0.22,
        val_ece=0.08,
        avg_loss=0.41,
        oos_sharpness=0.04,
    )
    dl_config = {
        "deploy_gate": {"enabled": False},
        "model_path_template": str(tmp_path / "{symbol}.pth"),
    }
    params = _training_params()
    prices = np.linspace(1.0, 2.0, 80)
    with (
        patch(
            "src.application.services.deep_learning.dl_symbol_train.train_model_walkforward",
            return_value=train_result,
        ),
        patch(
            "src.application.services.deep_learning.dl_symbol_train_success.evaluate_mini_deploy",
            return_value=(True, 0.58, 0.21),
        ),
        patch(
            "src.application.services.deep_learning.dl_symbol_train_success.save_model_checkpoint",
        ) as mock_save,
    ):
        stats, loss = run_symbol_training(
            "OTC_SPC",
            runtime,
            prices,
            dl_config,
            params,
            100,
            orch,
            granularity=60,
        )
    assert stats is train_result.norm_stats
    assert loss == 0.41
    assert runtime["session_trained"] is True
    assert mock_save.call_args.kwargs["granularity"] == 60


def test_run_symbol_training_wraps_export_sharpness_failure(tmp_path):
    orch = MagicMock()
    orch.infra = MagicMock(enabled=False)
    runtime = {
        "model": create_direction_model(arch="tcn"),
        "norm_stats": MagicMock(),
        "calibrator": MagicMock(method="platt"),
    }
    train_result = SimpleNamespace(
        norm_stats=MagicMock(),
        val_accuracy=0.61,
        calibrator=None,
        val_brier=0.22,
        val_ece=0.08,
        avg_loss=0.41,
        oos_sharpness=0.001,
        raw_sharpness=0.04,
    )
    dl_config = {
        "deploy_gate": {"enabled": False},
        "model_path_template": str(tmp_path / "{symbol}.pth"),
        "calibration": {"min_oos_sharpness": 0.01, "min_calibration_sharpness": 0.01},
    }
    params = _training_params()
    prices = np.linspace(1.0, 2.0, 80)
    with (
        patch(
            "src.application.services.deep_learning.dl_symbol_train.train_model_walkforward",
            return_value=train_result,
        ),
        patch(
            "src.application.services.deep_learning.dl_symbol_train_success.evaluate_mini_deploy",
            return_value=(True, 0.58, 0.21),
        ),
        patch(
            "src.application.services.deep_learning.dl_symbol_train_success.assert_export_sharpness_value",
            side_effect=RuntimeError("sharpness holdout 0.0010 < 0.0100"),
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
            granularity=60,
        )
    assert runtime["export_ok"] is False
    assert runtime["session_trained"] is False
    assert runtime["deploy_ok"] is False
