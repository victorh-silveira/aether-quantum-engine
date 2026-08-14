"""Cobertura de soft deploy no load de runtime."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from src.application.services.deep_learning.dl_calibration import CalibratorState
from src.application.services.deep_learning.dl_features import FEATURE_DIM
from src.application.services.deep_learning.dl_symbol_runtime import (
    _effective_deploy_ok,
    _persist_deploy_ok_flag,
    get_symbol_runtime,
)
from src.application.services.deep_learning.model import create_direction_model, fit_norm_stats


def test_effective_deploy_ok_soft_fallback():
    dl = {
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
        }
    }
    assert _effective_deploy_ok(stored_ok=False, val_accuracy=0.566, val_brier=0.250, dl_config=dl) is True
    assert _effective_deploy_ok(stored_ok=False, val_accuracy=0.566, val_brier=0.270, dl_config=dl) is False


def test_effective_deploy_ok_settle_bypasses_low_acc():
    dl = {
        "training_history_bars": 1333,
        "horizon_sweep": {
            "min_edge_vs_breakeven": 0.03,
            "min_settle_n": 16,
            "min_history_bars": 800,
            "payout_for_breakeven": 0.72,
        },
        "deploy_gate": {
            "enabled": True,
            "force_ok": False,
            "soft_min_val_accuracy": 0.53,
            "soft_max_brier": 0.26,
            "reject_majority_collapse": True,
            "max_label_call_frac_bias": 0.20,
            "min_minority_recall": 0.25,
        },
    }
    settings = {"deep_learning": dl, "data_handler": {"micro_granularity": 180}}
    payload = {
        "deploy_settlement_win_rate": 0.6774,
        "deploy_settlement_n": 31,
        "training_history_bars": 1333,
        "label_call_frac": 0.48,
        "pred_call_frac": 0.0,
        "minority_recall": 1.0,
    }
    assert (
        _effective_deploy_ok(
            stored_ok=True,
            val_accuracy=0.4941,
            val_brier=0.252,
            dl_config=dl,
            label_call_frac=0.48,
            pred_call_frac=0.0,
            minority_recall=1.0,
            checkpoint_payload=payload,
            settings=settings,
        )
        is True
    )
    assert (
        _effective_deploy_ok(
            stored_ok=True,
            val_accuracy=0.4941,
            val_brier=0.252,
            dl_config=dl,
            label_call_frac=0.48,
            pred_call_frac=0.0,
            minority_recall=1.0,
        )
        is False
    )


def test_persist_deploy_ok_flag(tmp_path: Path):
    path = tmp_path / "R_10.pth"
    torch.save({"deploy_ok": False, "val_accuracy": 0.56}, path)
    _persist_deploy_ok_flag(path, deploy_ok=True)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["deploy_ok"] is True


def test_get_symbol_runtime_promotes_soft_deploy(tmp_path: Path):
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
        "infra": {"triton": {"enabled": False}},
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
        patch(
            "src.application.services.deep_learning.dl_symbol_runtime.triton_enabled",
            return_value=False,
        ),
        patch(
            "src.application.services.deep_learning.dl_symbol_runtime._persist_deploy_ok_flag",
        ) as persist,
    ):
        path.write_bytes(b"x")
        runtime = get_symbol_runtime(orch, "R_10", orch.config["deep_learning"], {"lookback": 32, "arch": "tcn"})
    assert runtime["deploy_ok"] is True
    persist.assert_called_once()
