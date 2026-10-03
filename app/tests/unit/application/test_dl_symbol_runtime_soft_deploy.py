"""Cobertura de soft deploy no load de runtime."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from src.application.services.deep_learning.dl_calibration import CalibratorState
from src.application.services.deep_learning.dl_features import FEATURE_DIM
from src.application.services.deep_learning.dl_symbol_runtime import (
    get_symbol_runtime,
)
from src.application.services.deep_learning.model import create_direction_model, fit_norm_stats


def test_get_symbol_runtime_nao_promove_soft_deploy(tmp_path: Path):
    path = tmp_path / "R_10.pth"
    model = create_direction_model(arch="tcn", input_dim=FEATURE_DIM)
    stats = fit_norm_stats(np.zeros((1, 32, FEATURE_DIM), dtype=np.float32))
    loaded = (
        model,
        stats,
        1,
        CalibratorState(),
        32,
        0.566,
        0.250,
        0.1,
        False,
        0.0,
    )
    orch = MagicMock()
    orch.config = {
        "data_handler": {"micro_granularity": 180},
        "deep_learning": {
            "online_training": False,
            "allow_undeployed_inference": True,
            "deploy_gate": {
                "enabled": True,
                "force_ok": False,
                "max_brier": 0.22,
                "min_win_rate": 0.55,
                "mini_bars": 120,
                "max_eval_steps": 24,
                "min_trades": 2,
                "soft_min_val_accuracy": 0.53,
                "soft_max_brier": 0.26,
                "eval_relaxed_gating": True,
                "eval_call_threshold_cap": 0.65,
                "eval_put_threshold_floor": 0.01,
                "eval_call_threshold_default": 0.75,
                "eval_put_threshold_default": 0.25,
            },
        },
        "infra": {},
    }
    del orch._dl_runtime
    with (
        patch(
            "src.application.services.deep_learning.dl_symbol_runtime.resolve_dl_model_path",
            return_value=path,
        ),
        patch(
            "src.application.services.deep_learning.dl_symbol_runtime.load_model_checkpoint",
            return_value=loaded,
        ),
    ):
        path.write_bytes(b"x")
        runtime = get_symbol_runtime(orch, "R_10", orch.config["deep_learning"], {"lookback": 32, "arch": "tcn"})
    assert runtime["deploy_ok"] is False
