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


def test_get_symbol_runtime_requires_settlement_evidence_for_deploy():
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
        patch("pathlib.Path.is_file", return_value=False),
        patch("pathlib.Path.exists", return_value=False),
    ):
        runtime = get_symbol_runtime(orch, "R_10", dl_config, params)
    assert runtime["session_trained"] is True
    assert runtime["deploy_ok"] is False


def test_get_symbol_runtime_separates_training_from_deploy_quality():
    orch = MagicMock()
    orch.config = {"data_handler": {"granularity": 60}, "deep_learning": {}}
    orch._dl_runtime = {}
    dl_config = {"model_path_template": "data/dl/{symbol}.pth", "online_training": True}
    params = {"lookback": 48, "arch": "tcn"}
    with (
        patch(
            "src.application.services.deep_learning.dl_symbol_runtime.load_model_checkpoint",
            return_value=_loaded_checkpoint(deploy_ok=False, val_brier=0.9),
        ),
        patch("pathlib.Path.is_file", return_value=False),
        patch("pathlib.Path.exists", return_value=False),
    ):
        runtime = get_symbol_runtime(orch, "R_10", dl_config, params)
    assert runtime["session_trained"] is True
    assert runtime["checkpoint_loaded"] is True
    assert runtime["deploy_ok"] is False


def test_get_symbol_runtime_reuses_checkpoint_when_online_training_disabled():
    orch = MagicMock()
    orch.config = {"data_handler": {"granularity": 60}, "deep_learning": {}}
    orch._dl_runtime = {}
    dl_config = {
        "model_path_template": "data/dl/{symbol}.pth",
        "online_training": False,
        "deploy_gate": {"force_ok": False, "soft_max_brier": 0.26},
    }
    params = {"lookback": 48, "arch": "tcn"}
    with (
        patch(
            "src.application.services.deep_learning.dl_symbol_runtime.load_model_checkpoint",
            return_value=_loaded_checkpoint(deploy_ok=False, val_brier=0.25),
        ),
        patch("pathlib.Path.is_file", return_value=False),
        patch("pathlib.Path.exists", return_value=False),
    ):
        runtime = get_symbol_runtime(orch, "R_10", dl_config, params)
    assert runtime["session_trained"] is True
    assert runtime["deploy_ok"] is False


def test_get_symbol_runtime_force_ok_does_not_bypass_settlement_evidence():
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
        patch("pathlib.Path.is_file", return_value=False),
        patch("pathlib.Path.exists", return_value=False),
    ):
        runtime = get_symbol_runtime(orch, "R_10", dl_config, params)
    assert runtime["deploy_ok"] is False


def test_get_symbol_runtime_torch_load_failure_rejects_deploy():
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
    assert runtime["deploy_ok"] is False


def test_get_symbol_runtime_does_not_promote_legacy_payload():
    orch = MagicMock()
    orch.config = {"data_handler": {"granularity": 60}, "deep_learning": {}}
    orch._dl_runtime = {}
    dl_config = {"model_path_template": "data/dl/{symbol}.pth"}
    params = {"lookback": 48, "arch": "tcn"}
    payload = {
        "deploy_settlement_wilson_lcb": 0.8,
        "deploy_settlement_source": "broker_tick_audit",
        "label_call_frac": 0.5,
        "pred_call_frac": 0.5,
        "minority_recall": 0.8,
    }
    with (
        patch("pathlib.Path.exists", return_value=False),
        patch("pathlib.Path.is_file", return_value=True),
        patch("src.application.services.deep_learning.dl_symbol_runtime.torch.load", return_value=payload),
        patch(
            "src.application.services.deep_learning.dl_symbol_runtime.load_model_checkpoint",
            return_value=_loaded_checkpoint(deploy_ok=True),
        ),
    ):
        runtime = get_symbol_runtime(orch, "R_10", dl_config, params)
    assert runtime["deploy_ok"] is False


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


def test_checkpoint_fingerprint_tracks_exact_file_bytes(tmp_path):
    from src.application.services.deep_learning.dl_symbol_runtime import checkpoint_fingerprint

    path = tmp_path / "model.pth"
    assert checkpoint_fingerprint(path) is None
    path.write_bytes(b"first")
    first = checkpoint_fingerprint(path)
    assert first is not None and len(first) == 64
    path.write_bytes(b"second")
    assert checkpoint_fingerprint(path) != first
