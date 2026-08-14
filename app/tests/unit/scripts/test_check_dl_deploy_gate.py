from pathlib import Path
from unittest.mock import patch

from scripts.operations.check_dl_deploy_gate import evaluate_checkpoint


_COLLAPSE_OK = {"label_call_frac": 0.48, "pred_call_frac": 0.52, "minority_recall": 0.40}


def test_evaluate_checkpoint_rejects_low_acc(tmp_path: Path):
    path = tmp_path / "R_10.pth"
    path.write_bytes(b"x")
    with patch("torch.load", return_value={"val_accuracy": 0.52, "deploy_ok": True, **_COLLAPSE_OK}):
        ok, msg = evaluate_checkpoint(path, soft_min=0.53)
    assert ok is False
    assert "val_acc" in msg


def test_evaluate_checkpoint_accepts_settle_despite_low_acc(tmp_path: Path):
    path = tmp_path / "R_10.pth"
    path.write_bytes(b"x")
    settings = {
        "deep_learning": {
            "training_history_bars": 2000,
            "horizon_sweep": {
                "min_edge_vs_breakeven": 0.03,
                "min_settle_n": 16,
                "min_history_bars": 800,
                "payout_for_breakeven": 0.72,
            },
            "deploy_gate": {"soft_min_val_accuracy": 0.53},
        },
        "data_handler": {"micro_granularity": 180, "granularity": 7200},
    }
    payload = {
        "val_accuracy": 0.4667,
        "deploy_ok": False,
        "deploy_settlement_win_rate": 0.6923,
        "deploy_settlement_n": 26,
        "lookback": 720,
        "granularity": 180,
        **_COLLAPSE_OK,
    }
    with patch("torch.load", return_value=payload), patch("torch.save") as save_mock:
        ok, msg = evaluate_checkpoint(path, soft_min=0.53, settings=settings)
    assert ok is True
    assert "settle_ok" in msg
    save_mock.assert_called_once()
    assert payload["deploy_ok"] is True


def test_evaluate_checkpoint_rejects_deploy_false(tmp_path: Path):
    path = tmp_path / "R_10.pth"
    path.write_bytes(b"x")
    with patch(
        "torch.load",
        return_value={"val_accuracy": 0.60, "val_brier": 0.40, "deploy_ok": False, **_COLLAPSE_OK},
    ):
        ok, msg = evaluate_checkpoint(path, soft_min=0.53)
    assert ok is False
    assert "deploy_ok=false" in msg


def test_evaluate_checkpoint_rejects_missing_collapse_telemetry(tmp_path: Path):
    path = tmp_path / "R_10.pth"
    path.write_bytes(b"x")
    with patch("torch.load", return_value={"val_accuracy": 0.60, "val_brier": 0.20, "deploy_ok": True}):
        ok, msg = evaluate_checkpoint(path, soft_min=0.53)
    assert ok is False
    assert "telemetria de collapse ausente" in msg


def test_evaluate_checkpoint_soft_fallback_promotes_checkpoint(tmp_path: Path):
    path = tmp_path / "R_10.pth"
    path.write_bytes(b"x")
    payload = {"val_accuracy": 0.566, "val_brier": 0.250, "deploy_ok": False, **_COLLAPSE_OK}
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
                "reject_majority_collapse": True,
                "max_label_call_frac_bias": 0.20,
                "min_minority_recall": 0.25,
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
    with patch("torch.load", return_value={"val_accuracy": 0.55, "deploy_ok": True, **_COLLAPSE_OK}):
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
                "reject_majority_collapse": True,
                "max_label_call_frac_bias": 0.20,
                "min_minority_recall": 0.25,
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
            **_COLLAPSE_OK,
        },
    ):
        ok, msg = evaluate_checkpoint(path, soft_min=0.53, settings=settings)
    assert ok is False
    assert "lookback" in msg


def test_evaluate_checkpoint_rejects_stale_horizon(tmp_path: Path):
    path = tmp_path / "R_10.pth"
    path.write_bytes(b"x")
    settings = {
        "deep_learning": {
            "lookback": 480,
            "label_horizon_bars": 3,
            "train_timeframe": "micro",
            "deploy_gate": {"soft_min_val_accuracy": 0.53},
        },
        "data_handler": {"micro_granularity": 180, "granularity": 7200},
    }
    with patch(
        "torch.load",
        return_value={
            "val_accuracy": 0.55,
            "deploy_ok": True,
            "lookback": 480,
            "granularity": 180,
            "label_horizon_bars": 1,
            **_COLLAPSE_OK,
        },
    ):
        ok, msg = evaluate_checkpoint(path, soft_min=0.53, settings=settings)
    assert ok is False
    assert "label_horizon_bars" in msg


def test_evaluate_checkpoint_rejects_missing_horizon(tmp_path: Path):
    path = tmp_path / "R_10.pth"
    path.write_bytes(b"x")
    settings = {
        "deep_learning": {
            "lookback": 480,
            "label_horizon_bars": 3,
            "train_timeframe": "micro",
        },
        "data_handler": {"micro_granularity": 180, "granularity": 7200},
    }
    with patch(
        "torch.load",
        return_value={
            "val_accuracy": 0.55,
            "deploy_ok": True,
            "lookback": 480,
            "granularity": 180,
            **_COLLAPSE_OK,
        },
    ):
        ok, msg = evaluate_checkpoint(path, soft_min=0.53, settings=settings)
    assert ok is False
    assert "label_horizon_bars" in msg


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
        from scripts.operations.check_dl_deploy_gate import main

        assert main() == 1
