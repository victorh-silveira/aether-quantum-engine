from types import SimpleNamespace

import pytest
import torch

from src.application.services.deep_learning.dl_features import FEATURE_DIM
from src.application.services.deep_learning.dl_startup import (
    all_symbols_have_checkpoints,
    inference_startup_enabled,
    prepare_inference_run_loop,
    resolve_startup_fetch_bars,
)
from src.application.services.deep_learning.dl_training_gate import min_dl_history_len, resolve_train_ready_bars


@pytest.fixture(autouse=True)
def mock_torch_load(monkeypatch):
    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: {"feature_dim": FEATURE_DIM, "lookback": 48, "granularity": 120},
    )


def test_inference_startup_enabled_when_online_training_false():
    assert inference_startup_enabled({"online_training": False}) is True
    assert inference_startup_enabled({"online_training": True}) is False
    assert inference_startup_enabled({}) is True


def test_resolve_startup_fetch_bars_inference_mode(tmp_path, monkeypatch):
    repo = tmp_path
    dl_dir = repo / "data" / "dl"
    dl_dir.mkdir(parents=True)
    (dl_dir / "R_10.pth").write_bytes(b"x")
    config = {
        "symbols": ["R_10"],
        "data_handler": {"granularity": 120, "history_warmup_bars": 64},
        "deep_learning": {
            "online_training": False,
            "lookback": 48,
            "implied_vol_bars": 60,
        },
        "risk_management": {"params": {"duration": 60, "duration_unit": "s"}},
    }
    monkeypatch.setattr(
        "src.application.services.deep_learning.dl_startup.resolve_dl_model_path",
        lambda dl_cfg, symbol: dl_dir / f"{symbol}.pth",
    )
    bars, mode = resolve_startup_fetch_bars(config, ["R_10"])
    assert mode == "inferencia"
    assert bars == 256


def test_resolve_startup_fetch_bars_full_when_checkpoint_missing(tmp_path, monkeypatch):
    config = {
        "data_handler": {"fetch_count": 15616},
        "deep_learning": {"online_training": False},
    }
    monkeypatch.setattr(
        "src.application.services.deep_learning.dl_startup.resolve_dl_model_path",
        lambda dl_cfg, symbol: tmp_path / "missing.pth",
    )
    bars, mode = resolve_startup_fetch_bars(config, ["R_10"])
    assert mode == "treino"
    assert bars == 15616


def test_all_symbols_have_checkpoints(tmp_path, monkeypatch):
    path = tmp_path / "R_10.pth"
    path.write_bytes(b"1")
    monkeypatch.setattr(
        "src.application.services.deep_learning.dl_startup.resolve_dl_model_path",
        lambda _dl, symbol: tmp_path / f"{symbol}.pth",
    )
    monkeypatch.setattr(torch, "load", lambda *a, **kw: {"feature_dim": FEATURE_DIM})
    assert all_symbols_have_checkpoints(["R_10"], {}) is True
    assert all_symbols_have_checkpoints(["R_10", "R_50"], {}) is False


def test_prepare_inference_run_loop_marks_bootstrap_done(tmp_path, monkeypatch):
    (tmp_path / "R_10.pth").write_bytes(b"1")
    orch = SimpleNamespace(
        symbols=["R_10"],
        config={"deep_learning": {"online_training": False}, "data_handler": {}},
    )
    monkeypatch.setattr(
        "src.application.services.deep_learning.dl_startup.resolve_dl_model_path",
        lambda _dl, symbol: tmp_path / f"{symbol}.pth",
    )
    monkeypatch.setattr(torch, "load", lambda *a, **kw: {"feature_dim": FEATURE_DIM})
    assert prepare_inference_run_loop(orch) is True
    assert orch._dl_bootstrap_completed is True


def test_resolve_startup_fetch_bars_from_history_bars_when_training_mode(tmp_path, monkeypatch):
    config = {
        "data_handler": {"history_bars": 288, "history_warmup_bars": 32},
        "deep_learning": {"online_training": True},
    }
    bars, mode = resolve_startup_fetch_bars(config, ["R_10"])
    assert mode == "treino"
    assert bars == 320


def test_resolve_startup_fetch_bars_explicit_startup_fetch_bars(tmp_path, monkeypatch):
    (tmp_path / "R_10.pth").write_bytes(b"1")
    config = {
        "data_handler": {
            "startup_fetch_bars": 256,
            "history_warmup_bars": 0,
            "granularity": 3600,
            "micro_granularity": 120,
        },
        "deep_learning": {
            "online_training": False,
            "lookback": 48,
            "train_timeframe": "micro",
            "implied_vol_bars": 16,
        },
        "risk_management": {"params": {"duration": 2, "duration_unit": "m"}},
    }
    monkeypatch.setattr(
        "src.application.services.deep_learning.dl_startup.resolve_dl_model_path",
        lambda _dl, symbol: tmp_path / f"{symbol}.pth",
    )
    monkeypatch.setattr(
        torch,
        "load",
        lambda *a, **kw: {"feature_dim": FEATURE_DIM, "lookback": 48, "granularity": 120},
    )
    bars, mode = resolve_startup_fetch_bars(config, ["R_10"])
    assert mode == "inferencia"
    assert bars >= 256


def test_resolve_startup_fetch_bars_floors_below_lookback(tmp_path, monkeypatch):
    (tmp_path / "R_10.pth").write_bytes(b"1")
    config = {
        "data_handler": {"startup_fetch_bars": 512, "history_warmup_bars": 64, "micro_granularity": 120},
        "deep_learning": {
            "online_training": False,
            "lookback": 720,
            "train_timeframe": "micro",
            "implied_vol_bars": 60,
        },
        "risk_management": {"params": {"duration": 2, "duration_unit": "m"}},
    }
    monkeypatch.setattr(
        "src.application.services.deep_learning.dl_startup.resolve_dl_model_path",
        lambda _dl, symbol: tmp_path / f"{symbol}.pth",
    )
    monkeypatch.setattr(
        torch,
        "load",
        lambda *a, **kw: {"feature_dim": FEATURE_DIM, "lookback": 720, "granularity": 120},
    )
    bars, mode = resolve_startup_fetch_bars(config, ["R_10"])
    assert mode == "inferencia"
    assert bars >= 720 + 64
    assert bars > 512


def test_all_symbols_have_checkpoints_rejects_lookback_mismatch(tmp_path, monkeypatch):
    path = tmp_path / "R_10.pth"
    path.write_bytes(b"1")
    monkeypatch.setattr(
        "src.application.services.deep_learning.dl_startup.resolve_dl_model_path",
        lambda _dl, symbol: tmp_path / f"{symbol}.pth",
    )
    monkeypatch.setattr(
        torch,
        "load",
        lambda *a, **kw: {"feature_dim": FEATURE_DIM, "lookback": 128, "granularity": 60},
    )
    assert all_symbols_have_checkpoints(["R_10"], {"lookback": 72}, {"granularity": 600}) is False
    monkeypatch.setattr(
        torch,
        "load",
        lambda *a, **kw: {"feature_dim": FEATURE_DIM, "lookback": 72, "granularity": 60},
    )
    assert all_symbols_have_checkpoints(["R_10"], {"lookback": 72}, {"granularity": 600}) is False
    monkeypatch.setattr(
        torch,
        "load",
        lambda *a, **kw: {"feature_dim": FEATURE_DIM, "lookback": 72, "granularity": 600},
    )
    assert all_symbols_have_checkpoints(["R_10"], {"lookback": 72}, {"granularity": 600}) is True


def test_prepare_inference_run_loop_false_without_checkpoint(tmp_path, monkeypatch):
    orch = SimpleNamespace(
        symbols=["R_10"],
        config={"deep_learning": {"online_training": False}},
    )
    monkeypatch.setattr(
        "src.application.services.deep_learning.dl_startup.resolve_dl_model_path",
        lambda _dl, symbol: tmp_path / f"{symbol}.pth",
    )
    assert prepare_inference_run_loop(orch) is False
    assert not hasattr(orch, "_dl_bootstrap_completed")


def test_prepare_inference_run_loop_false_when_online_training_enabled():
    orch = SimpleNamespace(
        symbols=["R_10"],
        config={"deep_learning": {"online_training": True}, "data_handler": {}},
    )
    assert prepare_inference_run_loop(orch) is False
    assert not hasattr(orch, "_dl_bootstrap_completed")


def test_resolve_startup_fetch_bars_default_training_target():
    config = {"data_handler": {}, "deep_learning": {"online_training": True}}
    bars, mode = resolve_startup_fetch_bars(config, ["R_10"])
    assert mode == "treino"
    assert bars == 500


def test_min_dl_history_len_uses_inference_window_when_online_training_off():
    params = {
        "lookback": 48,
        "validation_bars": 96,
        "training_history_bars": 15552,
        "inference_history_bars": 128,
        "online_training": False,
    }
    assert min_dl_history_len(params) == 128


def test_min_dl_history_len_ignores_training_validation_bars_in_inference_mode():
    params = {
        "lookback": 48,
        "validation_bars": 3879,
        "inference_history_bars": 128,
        "online_training": False,
    }
    assert min_dl_history_len(params) == 128


def test_resolve_train_ready_bars_accepts_near_target_shortfall():
    params = {
        "lookback": 720,
        "validation_bars": 96,
        "training_history_bars": 2000,
        "inference_history_bars": 800,
        "train_history_shortfall_ratio": 0.95,
    }
    ok, want, soft = resolve_train_ready_bars(params, 1982)
    assert ok is True
    assert want == 2000
    assert soft is True
    ok2, _, soft2 = resolve_train_ready_bars(params, 1800)
    assert ok2 is False
    assert soft2 is False
    ok3, _, soft3 = resolve_train_ready_bars(params, 2000)
    assert ok3 is True
    assert soft3 is False
