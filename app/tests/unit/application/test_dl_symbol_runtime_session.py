from unittest.mock import MagicMock, patch

import numpy as np

from src.application.services.deep_learning.dl_symbol_runtime import get_symbol_runtime
from src.application.services.deep_learning.model import fit_norm_stats


def _loaded_checkpoint(*, deploy_ok: bool, val_brier: float = 0.22):
    lookback = 48
    stats = fit_norm_stats(np.zeros((1, lookback, 21), dtype=np.float32))
    model = MagicMock()
    return (
        model,
        stats,
        12345,
        MagicMock(),
        lookback,
        0.62,
        val_brier,
        0.1,
        deploy_ok,
        0.55,
    )


def test_get_symbol_runtime_marks_session_trained_when_deploy_ok_checkpoint():
    orch = MagicMock()
    orch.config = {"data_handler": {"granularity": 60}, "deep_learning": {}}
    orch._dl_runtime = {}
    dl_config = {"model_path_template": "data/dl/{symbol}.pth"}
    params = {"lookback": 48, "arch": "tcn"}
    with (
        patch(
            "src.application.services.deep_learning.dl_symbol_runtime.load_model_checkpoint",
            return_value=_loaded_checkpoint(deploy_ok=True),
        ),
        patch("pathlib.Path.exists", return_value=False),
    ):
        runtime = get_symbol_runtime(orch, "R_10", dl_config, params)
    assert runtime["session_trained"] is True
    assert runtime["deploy_ok"] is True


def test_get_symbol_runtime_keeps_session_untrained_without_deploy_ok():
    orch = MagicMock()
    orch.config = {"data_handler": {"granularity": 60}, "deep_learning": {}}
    orch._dl_runtime = {}
    dl_config = {"model_path_template": "data/dl/{symbol}.pth"}
    params = {"lookback": 48, "arch": "tcn"}
    with (
        patch(
            "src.application.services.deep_learning.dl_symbol_runtime.load_model_checkpoint",
            return_value=_loaded_checkpoint(deploy_ok=False, val_brier=0.9),
        ),
        patch("pathlib.Path.exists", return_value=False),
    ):
        runtime = get_symbol_runtime(orch, "R_10", dl_config, params)
    assert runtime["session_trained"] is False


def test_get_symbol_runtime_reuses_checkpoint_when_online_training_disabled():
    orch = MagicMock()
    orch.config = {"data_handler": {"granularity": 60}, "deep_learning": {}}
    orch._dl_runtime = {}
    dl_config = {
        "model_path_template": "data/dl/{symbol}.pth",
        "online_training": False,
        "deploy_gate": {"force_ok": False},
    }
    params = {"lookback": 48, "arch": "tcn"}
    with (
        patch(
            "src.application.services.deep_learning.dl_symbol_runtime.load_model_checkpoint",
            return_value=_loaded_checkpoint(deploy_ok=False, val_brier=0.40),
        ),
        patch("pathlib.Path.exists", return_value=False),
    ):
        runtime = get_symbol_runtime(orch, "R_10", dl_config, params)
    assert runtime["session_trained"] is True
    assert runtime["deploy_ok"] is False


def test_get_symbol_runtime_force_ok_overrides_deploy_flag():
    orch = MagicMock()
    orch.config = {"data_handler": {"granularity": 60}, "deep_learning": {}}
    orch._dl_runtime = {}
    dl_config = {
        "model_path_template": "data/dl/{symbol}.pth",
        "deploy_gate": {"force_ok": True},
    }
    params = {"lookback": 48, "arch": "tcn"}
    with (
        patch(
            "src.application.services.deep_learning.dl_symbol_runtime.load_model_checkpoint",
            return_value=_loaded_checkpoint(deploy_ok=False, val_brier=0.9),
        ),
        patch("pathlib.Path.exists", return_value=False),
    ):
        runtime = get_symbol_runtime(orch, "R_10", dl_config, params)
    assert runtime["deploy_ok"] is True


def test_get_symbol_runtime_exception_on_torch_load():
    orch = MagicMock()
    orch.config = {"data_handler": {"granularity": 60}, "deep_learning": {}}
    orch._dl_runtime = {}
    dl_config = {"model_path_template": "data/dl/{symbol}.pth"}
    params = {"lookback": 48, "arch": "tcn"}
    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("torch.load", side_effect=Exception("Corrupted file")),
        patch(
            "src.application.services.deep_learning.dl_symbol_runtime.load_model_checkpoint",
            return_value=_loaded_checkpoint(deploy_ok=True),
        ),
    ):
        runtime = get_symbol_runtime(orch, "R_10", dl_config, params)
    assert runtime["trained_granularity"] == 60
    assert runtime["deploy_ok"] is True


def test_get_symbol_runtime_discards_lookback_mismatch():
    orch = MagicMock()
    orch.config = {"data_handler": {"granularity": 600}, "deep_learning": {}}
    orch._dl_runtime = {}
    dl_config = {"model_path_template": "data/dl/{symbol}.pth", "online_training": False}
    params = {"lookback": 72, "arch": "tcn"}
    with patch(
        "src.application.services.deep_learning.dl_symbol_runtime.load_model_checkpoint",
        return_value=_loaded_checkpoint(deploy_ok=True),
    ):
        runtime = get_symbol_runtime(orch, "R_10", dl_config, params)
    assert runtime["session_trained"] is False
    assert runtime["lookback"] == 72
    assert runtime["trained_granularity"] == 600


def test_get_symbol_runtime_logs_retrain_when_online_training_and_mismatch():
    orch = MagicMock()
    orch.config = {"data_handler": {"granularity": 600}, "deep_learning": {}}
    orch._dl_runtime = {}
    dl_config = {"model_path_template": "data/dl/{symbol}.pth", "online_training": True}
    params = {"lookback": 72, "arch": "tcn"}
    with patch(
        "src.application.services.deep_learning.dl_symbol_runtime.load_model_checkpoint",
        return_value=_loaded_checkpoint(deploy_ok=True),
    ):
        runtime = get_symbol_runtime(orch, "R_10", dl_config, params)
    assert runtime["session_trained"] is False
    assert runtime["lookback"] == 72
