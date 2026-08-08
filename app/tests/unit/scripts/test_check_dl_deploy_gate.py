"""Gate de deploy/ACC apos treino DL."""

from pathlib import Path
from unittest.mock import patch

from scripts.operations.check_dl_deploy_gate import evaluate_checkpoint, main


def test_evaluate_checkpoint_rejects_low_acc(tmp_path: Path):
    path = tmp_path / "R_10.pth"
    path.write_bytes(b"x")
    with patch("torch.load", return_value={"val_accuracy": 0.52, "deploy_ok": True}):
        ok, msg = evaluate_checkpoint(path, soft_min=0.53)
    assert ok is False
    assert "val_acc" in msg


def test_evaluate_checkpoint_rejects_deploy_false(tmp_path: Path):
    path = tmp_path / "R_10.pth"
    path.write_bytes(b"x")
    with patch(
        "torch.load",
        return_value={"val_accuracy": 0.60, "val_brier": 0.40, "deploy_ok": False},
    ):
        ok, msg = evaluate_checkpoint(path, soft_min=0.53)
    assert ok is False
    assert "deploy_ok=false" in msg


def test_evaluate_checkpoint_soft_fallback_promotes_checkpoint(tmp_path: Path):
    path = tmp_path / "R_10.pth"
    path.write_bytes(b"x")
    payload = {"val_accuracy": 0.566, "val_brier": 0.250, "deploy_ok": False}
    settings = {
        "deep_learning": {
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
    }
    with patch("torch.load", return_value=payload), patch("torch.save") as save_mock:
        ok, msg = evaluate_checkpoint(path, soft_min=0.53, settings=settings)
    assert ok is True
    assert "soft fallback" in msg
    save_mock.assert_called_once()
    assert payload["deploy_ok"] is True


def test_evaluate_checkpoint_accepts_senior(tmp_path: Path):
    path = tmp_path / "R_10.pth"
    path.write_bytes(b"x")
    with patch("torch.load", return_value={"val_accuracy": 0.55, "deploy_ok": True}):
        ok, msg = evaluate_checkpoint(path, soft_min=0.53)
    assert ok is True
    assert "deploy_ok=true" in msg


def test_evaluate_checkpoint_rejects_stale_geometry(tmp_path: Path):
    path = tmp_path / "R_10.pth"
    path.write_bytes(b"x")
    settings = {
        "deep_learning": {
            "lookback": 720,
            "train_timeframe": "micro",
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
        "data_handler": {"micro_granularity": 60, "granularity": 300},
    }
    with patch(
        "torch.load",
        return_value={
            "val_accuracy": 0.55,
            "deploy_ok": True,
            "lookback": 360,
            "granularity": 120,
        },
    ):
        ok, msg = evaluate_checkpoint(path, soft_min=0.53, settings=settings)
    assert ok is False
    assert "lookback" in msg


def test_main_fails_when_checkpoint_missing(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("sys.argv", ["check_dl_deploy_gate.py", "--symbols", "R_10"])
    with (
        patch(
            "scripts.operations.check_dl_deploy_gate._checkpoint_paths",
            return_value=[tmp_path / "missing.pth"],
        ),
        patch("scripts.operations.check_dl_deploy_gate._load_settings", return_value={}),
        patch("scripts.operations.check_dl_deploy_gate._soft_min_acc", return_value=0.53),
    ):
        assert main() == 1
