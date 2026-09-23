"""Cobertura residual: overwrite de checkpoint apos treino."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from src.application.services.deep_learning.dl_symbol_train_success import apply_successful_symbol_train


def test_apply_successful_symbol_train_deploy_warning(tmp_path):
    ckpt = tmp_path / "R_10.pth"
    torch.save({"deploy_ok": True, "val_brier": 0.22}, ckpt)
    runtime = {"val_accuracy": 0.56, "val_brier": 0.27}
    train_result = SimpleNamespace(
        norm_stats=MagicMock(),
        val_accuracy=0.56,
        val_brier=0.27,
        calibrator=None,
        val_ece=0.1,
        avg_loss=0.4,
        epochs_ran=40,
        oos_sharpness=0.05,
    )
    gate_cfg = {"enabled": True, "soft_min_val_accuracy": 0.53, "soft_max_brier": 0.26}
    dl_config = {"model_path_template": "x/{symbol}.pth", "deploy_gate": gate_cfg}
    orch = MagicMock()
    with (
        patch(
            "src.application.services.deep_learning.dl_symbol_train_success.evaluate_mini_deploy",
            return_value=(False, 0.5, 0.27),
        ),
        patch(
            "src.application.services.deep_learning.dl_symbol_train_success.save_model_checkpoint",
        ) as save_ckpt,
        patch(
            "src.application.services.deep_learning.dl_symbol_train_success.schedule_model_upload",
        ),
        patch(
            "src.application.services.deep_learning.dl_symbol_train_success.resolve_dl_model_path",
            return_value=ckpt,
        ),
        patch(
            "src.application.services.deep_learning.dl_symbol_train_success.resolve_deploy_ok",
            return_value=False,
        ),
    ):
        apply_successful_symbol_train(
            "R_10",
            runtime,
            train_result,
            orch=orch,
            model=MagicMock(),
            prices=np.linspace(1.0, 2.0, 80),
            norm_stats=train_result.norm_stats,
            params={"lookback": 32, "arch": "tcn"},
            dl_config=dl_config,
            gate_cfg=gate_cfg,
            candle_epoch_value=1,
            granularity=60,
            level=logging.INFO,
            started=0.0,
        )
    assert runtime.get("checkpoint_preserved") is False
    assert runtime.get("session_trained") is False
    assert runtime.get("export_ok") is False
    assert save_ckpt.called
    assert save_ckpt.call_args.kwargs.get("deploy_ok") is False


def test_apply_successful_symbol_train_overwrites_previous_checkpoint(tmp_path):
    ckpt = tmp_path / "R_10.pth"
    torch.save(
        {
            "deploy_ok": True,
            "val_accuracy": 0.556,
            "val_brier": 0.24,
            "label_call_frac": 0.44,
            "pred_call_frac": 0.66,
            "minority_recall": 0.38,
        },
        ckpt,
    )
    runtime = {"val_accuracy": 0.50, "val_brier": 0.27}
    train_result = SimpleNamespace(
        norm_stats=MagicMock(),
        val_accuracy=0.50,
        val_brier=0.27,
        calibrator=None,
        val_ece=0.1,
        avg_loss=0.4,
        epochs_ran=40,
        oos_sharpness=0.05,
        label_call_frac=0.49,
        pred_call_frac=0.36,
        minority_recall=0.65,
    )
    gate_cfg = {"enabled": True, "soft_min_val_accuracy": 0.53, "soft_max_brier": 0.26}
    dl_config = {"model_path_template": "x/{symbol}.pth", "deploy_gate": gate_cfg}
    orch = MagicMock()
    with (
        patch(
            "src.application.services.deep_learning.dl_symbol_train_success.evaluate_mini_deploy",
            return_value=(False, 0.5, 0.27),
        ),
        patch(
            "src.application.services.deep_learning.dl_symbol_train_success.save_model_checkpoint",
        ) as save_ckpt,
        patch(
            "src.application.services.deep_learning.dl_symbol_train_success.schedule_model_upload",
        ),
        patch(
            "src.application.services.deep_learning.dl_symbol_train_success.resolve_dl_model_path",
            return_value=ckpt,
        ),
    ):
        apply_successful_symbol_train(
            "R_10",
            runtime,
            train_result,
            orch=orch,
            model=MagicMock(),
            prices=np.linspace(1.0, 2.0, 80),
            norm_stats=train_result.norm_stats,
            params={"lookback": 32, "arch": "tcn"},
            dl_config=dl_config,
            gate_cfg=gate_cfg,
            candle_epoch_value=1,
            granularity=120,
            level=logging.INFO,
            started=0.0,
        )
    assert runtime.get("checkpoint_preserved") is False
    assert runtime.get("export_ok") is False
    assert save_ckpt.called
    assert save_ckpt.call_args.kwargs.get("deploy_ok") is False


def test_apply_successful_symbol_train_nao_promove_atalho_do_sweep(tmp_path):
    ckpt = tmp_path / "1HZ75V.pth"
    runtime = {"val_accuracy": 0.51, "val_brier": 0.27}
    train_result = SimpleNamespace(
        norm_stats=MagicMock(),
        val_accuracy=0.51,
        val_brier=0.27,
        calibrator=None,
        val_ece=0.1,
        avg_loss=0.4,
        epochs_ran=40,
        oos_sharpness=0.05,
        label_call_frac=0.5,
        pred_call_frac=0.5,
        minority_recall=0.5,
    )
    gate_cfg = {"enabled": True, "soft_min_val_accuracy": 0.53, "soft_max_brier": 0.28}
    dl_config = {"model_path_template": "x/{symbol}.pth", "deploy_gate": gate_cfg}
    orch = MagicMock()
    orch.config = {
        "deep_learning": {
            "horizon_sweep": {
                "min_edge_vs_breakeven": 0.03,
                "min_settle_n": 16,
                "min_history_bars": 800,
            },
            "training_history_bars": 2000,
        },
        "risk_management": {"params": {"payout_estimate": 0.85}},
    }

    def _mini_deploy(_orch, _symbol, _model, _prices, _norm, rt, *_a, **_k):
        rt["deploy_settlement_win_rate"] = 0.59
        rt["deploy_settlement_n"] = 24
        return False, 0.50, 0.27

    with (
        patch(
            "src.application.services.deep_learning.dl_symbol_train_success.evaluate_mini_deploy",
            side_effect=_mini_deploy,
        ),
        patch(
            "src.application.services.deep_learning.dl_symbol_train_success.save_model_checkpoint",
        ) as save_ckpt,
        patch(
            "src.application.services.deep_learning.dl_symbol_train_success.schedule_model_upload",
        ),
        patch(
            "src.application.services.deep_learning.dl_symbol_train_success.resolve_dl_model_path",
            return_value=ckpt,
        ),
        patch(
            "src.application.services.deep_learning.dl_symbol_train_success.resolve_deploy_ok",
            return_value=False,
        ),
    ):
        apply_successful_symbol_train(
            "1HZ75V",
            runtime,
            train_result,
            orch=orch,
            model=MagicMock(),
            prices=np.linspace(1.0, 2.0, 80),
            norm_stats=train_result.norm_stats,
            params={
                "lookback": 30,
                "arch": "tcn",
                "training_history_bars": 2000,
                "label_horizon_bars": 1,
            },
            dl_config=dl_config,
            gate_cfg=gate_cfg,
            candle_epoch_value=1,
            granularity=300,
            level=logging.INFO,
            started=0.0,
        )
    assert runtime.get("deploy_ok") is False
    assert runtime.get("export_ok") is False
    assert runtime.get("session_trained") is False
    assert save_ckpt.called
    assert save_ckpt.call_args.kwargs.get("deploy_ok") is False


def test_apply_successful_symbol_train_logs_horizon(tmp_path):
    ckpt = tmp_path / "R_10.pth"
    runtime = {"val_accuracy": 0.56, "val_brier": 0.22}
    train_result = SimpleNamespace(
        norm_stats=MagicMock(),
        val_accuracy=0.56,
        val_brier=0.22,
        calibrator=None,
        val_ece=0.1,
        avg_loss=0.4,
        epochs_ran=10,
        oos_sharpness=0.05,
        label_call_frac=0.5,
        pred_call_frac=0.5,
        minority_recall=0.5,
    )
    gate_cfg = {"enabled": True, "soft_min_val_accuracy": 0.53, "soft_max_brier": 0.26}
    dl_config = {"model_path_template": str(tmp_path / "{symbol}.pth"), "deploy_gate": gate_cfg}
    orch = MagicMock()
    orch.config = {"risk_management": {"params": {"contract_duration_seconds": 120}}}
    with (
        patch(
            "src.application.services.deep_learning.dl_symbol_train_success.evaluate_mini_deploy",
            return_value=(True, 0.55, 0.20),
        ),
        patch(
            "src.application.services.deep_learning.dl_symbol_train_success.save_model_checkpoint",
        ),
        patch(
            "src.application.services.deep_learning.dl_symbol_train_success.schedule_model_upload",
        ),
        patch(
            "src.application.services.deep_learning.dl_symbol_train_success.resolve_dl_model_path",
            return_value=ckpt,
        ),
        patch(
            "src.application.services.deep_learning.dl_symbol_train_success.resolve_deploy_ok",
            return_value=True,
        ),
    ):
        apply_successful_symbol_train(
            "R_10",
            runtime,
            train_result,
            orch=orch,
            model=MagicMock(),
            prices=np.linspace(1.0, 2.0, 80),
            norm_stats=train_result.norm_stats,
            params={"lookback": 32, "arch": "tcn", "label_horizon_bars": 1},
            dl_config=dl_config,
            gate_cfg=gate_cfg,
            candle_epoch_value=1,
            granularity=120,
            level=logging.INFO,
            started=0.0,
        )
    assert runtime.get("session_trained") is True
    assert runtime.get("export_ok") is True

    runtime_fail = {"val_accuracy": 0.56, "val_brier": 0.20}
    train_res_fail = SimpleNamespace(
        norm_stats=MagicMock(),
        val_accuracy=0.56,
        val_brier=0.20,
        calibrator=None,
        val_ece=0.1,
        avg_loss=0.4,
        epochs_ran=40,
        oos_sharpness=0.01,
        raw_sharpness=0.01,
    )
    with (
        patch(
            "src.application.services.deep_learning.dl_symbol_train_success.evaluate_mini_deploy",
            return_value=(True, 0.56, 0.20),
        ),
        patch("src.application.services.deep_learning.dl_symbol_train_success.save_model_checkpoint"),
        patch("src.application.services.deep_learning.dl_symbol_train_success.schedule_model_upload"),
        patch(
            "src.application.services.deep_learning.dl_symbol_train_success.resolve_dl_model_path", return_value=ckpt
        ),
        patch("src.application.services.deep_learning.dl_symbol_train_success.resolve_deploy_ok", return_value=True),
        patch(
            "src.application.services.deep_learning.dl_symbol_train_success.assert_export_sharpness_value",
            side_effect=RuntimeError("sharp"),
        ),
    ):
        apply_successful_symbol_train(
            "R_10",
            runtime_fail,
            train_res_fail,
            orch=orch,
            model=MagicMock(),
            prices=np.linspace(1.0, 2.0, 80),
            norm_stats=train_res_fail.norm_stats,
            params={"lookback": 32, "arch": "tcn", "label_horizon_bars": 1},
            dl_config={"deploy_gate": {"force_ok": False}, "sharpness": {"min_oos_sharpness": 0.05}},
            gate_cfg={"force_ok": False},
            candle_epoch_value=1,
            granularity=120,
            level=logging.INFO,
            started=0.0,
        )
    assert runtime_fail.get("deploy_ok") is False
