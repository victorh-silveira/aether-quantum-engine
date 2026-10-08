"""Verificacao tecnica dos checkpoints apos o treino."""

import sys
from unittest.mock import patch

import pytest
import torch

from scripts.operations.check_dl_checkpoint import _checkpoint_paths, _expected_geometry, evaluate_checkpoint, main
from src.application.services.deep_learning.dl_features import FEATURE_DIM
from src.application.services.deep_learning.model import create_direction_model


@pytest.fixture
def settings():
    return {
        "deep_learning": {
            "lookback": 32,
            "label_horizon_bars": 1,
            "label_mode": "spot_forward",
            "train_timeframe": "micro",
            "model_path_template": "data/dl/{symbol}.pth",
        },
        "data_handler": {"micro_granularity": 300, "granularity": 86400},
        "symbols": ["1HZ75V"],
    }


def test_expected_geometry_and_paths(settings, tmp_path):
    from scripts.operations import check_dl_checkpoint as gate

    assert _expected_geometry(settings)["granularity"] == 300
    settings["deep_learning"]["train_timeframe"] = "macro"
    assert _expected_geometry(settings)["granularity"] == 86400
    with patch.object(gate, "REPO_ROOT", tmp_path):
        assert _checkpoint_paths(settings, ["1HZ75V"]) == [tmp_path / "data/dl/1HZ75V.pth"]


def test_evaluate_checkpoint_requires_integrity_and_geometry(settings, tmp_path):
    path = tmp_path / "model.pth"
    assert evaluate_checkpoint(path, settings=settings)[0] is False
    path.write_bytes(b"broken")
    assert evaluate_checkpoint(path, settings=settings)[0] is False
    torch.save([], path)
    assert evaluate_checkpoint(path, settings=settings)[0] is False
    payload = {
        **_expected_geometry(settings),
        "state_dict": create_direction_model(arch="tcn").state_dict(),
        "norm_mean": torch.zeros(FEATURE_DIM),
        "norm_std": torch.ones(FEATURE_DIM),
    }
    torch.save(payload, path)
    assert evaluate_checkpoint(path, settings=settings)[0] is True
    payload["lookback"] = 30
    torch.save(payload, path)
    assert evaluate_checkpoint(path, settings=settings)[0] is False


def test_main_reports_missing_and_compatible_checkpoint(settings, tmp_path):
    from scripts.operations import check_dl_checkpoint as gate

    with (
        patch.object(gate, "REPO_ROOT", tmp_path),
        patch.object(gate, "_load_settings", return_value=settings),
        patch.object(sys, "argv", ["check_dl_checkpoint.py"]),
    ):
        assert main() == 1
        path = tmp_path / "data/dl/1HZ75V.pth"
        path.parent.mkdir(parents=True)
        torch.save(
            {
                **_expected_geometry(settings),
                "state_dict": create_direction_model(arch="tcn").state_dict(),
                "norm_mean": torch.zeros(FEATURE_DIM),
                "norm_std": torch.ones(FEATURE_DIM),
            },
            path,
        )
        assert main() == 0
