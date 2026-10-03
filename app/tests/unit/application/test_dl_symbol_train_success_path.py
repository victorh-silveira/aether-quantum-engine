"""Integra treino temporal e persistencia do checkpoint tecnico."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from src.application.services.deep_learning.dl_symbol_train import run_symbol_training
from src.application.services.deep_learning.model import create_direction_model
from tests.unit.application.test_dl_training_progress import _training_params


def _setup(tmp_path):
    orch = MagicMock()
    orch.infra = MagicMock(enabled=False)
    runtime = {"model": create_direction_model(arch="tcn"), "norm_stats": MagicMock()}
    result = SimpleNamespace(
        norm_stats=MagicMock(),
        val_accuracy=0.61,
        calibrator=None,
        val_brier=0.22,
        val_ece=0.08,
        avg_loss=0.41,
        epochs_ran=5,
    )
    config = {"model_path_template": str(tmp_path / "{symbol}.pth")}
    return orch, runtime, result, config


def test_run_symbol_training_persists_technical_checkpoint(tmp_path):
    orch, runtime, result, config = _setup(tmp_path)
    with (
        patch("src.application.services.deep_learning.dl_symbol_train.train_model_walkforward", return_value=result),
        patch("src.application.services.deep_learning.dl_symbol_train_success.save_model_checkpoint") as saved,
    ):
        stats, loss = run_symbol_training(
            "R_10",
            runtime,
            np.linspace(1.0, 2.0, 80),
            config,
            _training_params(),
            100,
            orch,
            granularity=60,
        )
    assert stats is result.norm_stats
    assert loss == 0.41
    assert runtime["session_trained"] is True
    assert runtime["deploy_ok"] is False
    assert saved.call_args.kwargs["granularity"] == 60


def test_run_symbol_training_fails_closed_on_export_error(tmp_path):
    orch, runtime, result, config = _setup(tmp_path)
    with (
        patch("src.application.services.deep_learning.dl_symbol_train.train_model_walkforward", return_value=result),
        patch(
            "src.application.services.deep_learning.dl_symbol_train_success.save_model_checkpoint",
            side_effect=OSError("disk full"),
        ),
    ):
        run_symbol_training(
            "R_10",
            runtime,
            np.linspace(1.0, 2.0, 80),
            config,
            _training_params(),
            100,
            orch,
            granularity=60,
        )
    assert runtime["export_ok"] is False
    assert runtime["session_trained"] is False
    assert runtime["deploy_ok"] is False
