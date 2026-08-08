import pytest
import torch

from src.application.services.deep_learning.dl_features import FEATURE_DIM
from src.application.services.deep_learning.dl_startup import (
    all_symbols_have_checkpoints,
    resolve_startup_fetch_bars,
)
from src.application.services.deep_learning.dl_training_gate import resolve_train_ready_bars


@pytest.fixture(autouse=True)
def mock_torch_load(monkeypatch):
    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: {"feature_dim": FEATURE_DIM, "lookback": 48, "granularity": 120},
    )


def test_resolve_startup_fetch_bars_honors_startup_fetch_bars(tmp_path, monkeypatch):
    dl_dir = tmp_path / "data" / "dl"
    dl_dir.mkdir(parents=True)
    (dl_dir / "R_10.pth").write_bytes(b"x")
    config = {
        "symbols": ["R_10"],
        "data_handler": {"granularity": 120, "history_warmup_bars": 64, "startup_fetch_bars": 300},
        "deep_learning": {"online_training": False, "lookback": 48, "implied_vol_bars": 60},
        "risk_management": {"params": {"duration": 60, "duration_unit": "s"}},
    }
    monkeypatch.setattr(
        "src.application.services.deep_learning.dl_startup.resolve_dl_model_path",
        lambda dl_cfg, symbol: dl_dir / f"{symbol}.pth",
    )
    bars, mode = resolve_startup_fetch_bars(config, ["R_10"])
    assert mode == "inferencia"
    assert bars == 300


def test_resolve_train_ready_bars_invalid_shortfall_ratio_defaults():
    params = {
        "lookback": 720,
        "validation_bars": 96,
        "training_history_bars": 2000,
        "train_history_shortfall_ratio": "bad",
    }
    ok, want, soft = resolve_train_ready_bars(params, 1982)
    assert ok is True
    assert want == 2000
    assert soft is True


def test_all_symbols_have_checkpoints_incompatible_or_error(tmp_path, monkeypatch):
    path = tmp_path / "R_10.pth"
    path.write_bytes(b"1")
    monkeypatch.setattr(
        "src.application.services.deep_learning.dl_startup.resolve_dl_model_path",
        lambda _dl, symbol: tmp_path / f"{symbol}.pth",
    )
    monkeypatch.setattr(torch, "load", lambda *a, **kw: {"feature_dim": 26})
    assert all_symbols_have_checkpoints(["R_10"], {}) is False

    def raise_err(*a, **kw):
        raise ValueError("Load error")

    monkeypatch.setattr(torch, "load", raise_err)
    assert all_symbols_have_checkpoints(["R_10"], {}) is False
