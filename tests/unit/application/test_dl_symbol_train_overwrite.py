"""Checkpoint local substitui o anterior sem promocao estatistica."""

import logging
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from src.application.services.deep_learning.dl_symbol_train_success import apply_successful_symbol_train


def test_training_persists_technical_checkpoint_without_deploy(tmp_path, caplog):
    result = SimpleNamespace(
        norm_stats=MagicMock(),
        val_accuracy=0.56,
        val_brier=0.27,
        val_ece=0.1,
        calibrator=None,
        avg_loss=0.4,
        epochs_ran=40,
    )
    runtime = {}
    orch = MagicMock()
    orch.config = {"risk_management": {"params": {"duration": 5, "duration_unit": "m"}}}
    with (
        patch("src.application.services.deep_learning.dl_symbol_train_success.save_model_checkpoint") as saved,
        patch(
            "src.application.services.deep_learning.dl_symbol_train_success.resolve_dl_model_path",
            return_value=tmp_path / "R_10.pth",
        ),
        caplog.at_level(logging.INFO),
    ):
        stats, loss = apply_successful_symbol_train(
            "R_10",
            runtime,
            result,
            orch=orch,
            model=MagicMock(),
            prices=np.linspace(1.0, 2.0, 80),
            norm_stats=result.norm_stats,
            params={"lookback": 32, "arch": "tcn", "label_horizon_bars": 1},
            dl_config={},
            candle_epoch_value=1,
            granularity=300,
            level=logging.INFO,
            started=time.monotonic(),
        )
    assert stats is result.norm_stats
    assert loss == 0.4
    assert runtime["checkpoint_loaded"] is True
    assert runtime["session_trained"] is True
    assert runtime["deploy_ok"] is False
    assert runtime["checkpoint_preserved"] is False
    assert saved.call_args.kwargs["deploy_ok"] is False
    assert "teto de 1%" in caplog.text
    assert "gap=0s" in caplog.text
