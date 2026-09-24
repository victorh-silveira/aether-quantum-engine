from pathlib import Path
from unittest.mock import patch

from scripts.operations.check_dl_deploy_gate import evaluate_checkpoint


_COLLAPSE_OK = {"label_call_frac": 0.48, "pred_call_frac": 0.52, "minority_recall": 0.40}


_STRICT_GATE = {
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


def test_evaluate_checkpoint_rejects_low_acc(tmp_path: Path):
    path = tmp_path / "R_10.pth"
    path.write_bytes(b"x")
    settings = {"deep_learning": {"deploy_gate": dict(_STRICT_GATE)}}
    with patch("torch.load", return_value={"val_accuracy": 0.52, "deploy_ok": True, **_COLLAPSE_OK}):
        ok, msg = evaluate_checkpoint(path, soft_min=0.53, settings=settings)
    assert ok is False
    assert "val_acc" in msg


def test_evaluate_checkpoint_rejects_old_label_contract(tmp_path: Path):
    path = tmp_path / "1HZ75V.pth"
    path.write_bytes(b"x")
    settings = {"deep_learning": {"label_mode": "spot_forward", "deploy_gate": dict(_STRICT_GATE)}}
    with patch("torch.load", return_value={"label_mode": "quantum_multi_barrier", "deploy_ok": True}):
        ok, reason = evaluate_checkpoint(path, soft_min=0.0, settings=settings)
    assert ok is False
    assert "label_mode" in reason


def test_evaluate_checkpoint_rejeita_acc_abaixo_do_piso_anticolapso(tmp_path: Path):
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
            "deploy_gate": dict(_STRICT_GATE),
        },
        "data_handler": {"micro_granularity": 180, "granularity": 7200},
    }
    payload = {
        "val_accuracy": 0.51,
        "deploy_ok": True,
        "deploy_settlement_win_rate": 0.6923,
        "deploy_settlement_n": 26,
        "deploy_settlement_wilson_lcb": 0.60,
        "lookback": 720,
        "granularity": 180,
        **_COLLAPSE_OK,
    }
    with patch("torch.load", return_value=payload), patch("torch.save") as save_mock:
        ok, msg = evaluate_checkpoint(path, soft_min=0.53, settings=settings)
    assert ok is False
    assert "val_acc" in msg
    save_mock.assert_not_called()


def test_evaluate_checkpoint_rejects_deploy_false(tmp_path: Path):
    path = tmp_path / "R_10.pth"
    path.write_bytes(b"x")
    settings = {"deep_learning": {"deploy_gate": dict(_STRICT_GATE)}}
    with patch(
        "torch.load",
        return_value={"val_accuracy": 0.60, "val_brier": 0.40, "deploy_ok": False, **_COLLAPSE_OK},
    ):
        ok, msg = evaluate_checkpoint(path, soft_min=0.53, settings=settings)
    assert ok is False
    assert "deploy_ok=false" in msg


def test_evaluate_checkpoint_rejects_missing_collapse_telemetry(tmp_path: Path):
    path = tmp_path / "R_10.pth"
    path.write_bytes(b"x")
    settings = {"deep_learning": {"deploy_gate": dict(_STRICT_GATE)}}
    with patch("torch.load", return_value={"val_accuracy": 0.60, "val_brier": 0.20, "deploy_ok": True}):
        ok, msg = evaluate_checkpoint(path, soft_min=0.53, settings=settings)
    assert ok is False
    assert "telemetria de collapse ausente" in msg


def test_evaluate_checkpoint_rejeita_soft_fallback_sem_settlement(tmp_path: Path):
    path = tmp_path / "R_10.pth"
    path.write_bytes(b"x")
    payload = {"val_accuracy": 0.566, "val_brier": 0.250, "deploy_ok": False, **_COLLAPSE_OK}
    settings = {"deep_learning": {"deploy_gate": dict(_STRICT_GATE)}}
    with patch("torch.load", return_value=payload), patch("torch.save") as save_mock:
        ok, msg = evaluate_checkpoint(path, soft_min=0.53, settings=settings)
    assert ok is False
    assert "sem evidencia OOS" in msg
    save_mock.assert_not_called()
    assert payload["deploy_ok"] is False


def test_evaluate_checkpoint_accepts_senior(tmp_path: Path):
    path = tmp_path / "R_10.pth"
    path.write_bytes(b"x")
    settings = {"deep_learning": {"deploy_gate": dict(_STRICT_GATE)}}
    with patch(
        "torch.load",
        return_value={
            "val_accuracy": 0.55,
            "deploy_ok": True,
            "deploy_settlement_wilson_lcb": 0.60,
            "deploy_settlement_source": "broker_tick_audit",
            **_COLLAPSE_OK,
        },
    ):
        ok, msg = evaluate_checkpoint(path, soft_min=0.53, settings=settings)
    assert ok is True
    assert "deploy_ok=true" in msg


def test_evaluate_checkpoint_rejects_m5_proxy_even_with_high_lcb(tmp_path: Path):
    path = tmp_path / "R_10.pth"
    path.write_bytes(b"x")
    settings = {"deep_learning": {"deploy_gate": dict(_STRICT_GATE)}}
    with patch(
        "torch.load",
        return_value={
            "val_accuracy": 0.60,
            "deploy_ok": True,
            "deploy_settlement_wilson_lcb": 0.70,
            "deploy_settlement_source": "m5_close_proxy",
            **_COLLAPSE_OK,
        },
    ):
        ok, msg = evaluate_checkpoint(path, soft_min=0.53, settings=settings)
    assert ok is False
    assert "apenas proxy" in msg


def test_evaluate_checkpoint_rejects_stale_geometry(tmp_path: Path):
    path = tmp_path / "R_10.pth"
    path.write_bytes(b"x")
    settings = {
        "deep_learning": {
            "lookback": 720,
            "train_timeframe": "micro",
            "deploy_gate": dict(_STRICT_GATE),
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
            "deploy_gate": dict(_STRICT_GATE),
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


def test_evaluate_checkpoint_force_ok_nao_promove_modelo_fraco(tmp_path: Path):
    path = tmp_path / "R_10.pth"
    path.write_bytes(b"x")
    payload = {"val_accuracy": 0.40, "val_brier": 0.50, "deploy_ok": False, **_COLLAPSE_OK}
    settings = {
        "deep_learning": {
            "deploy_gate": {**_STRICT_GATE, "force_ok": True, "soft_min_val_accuracy": 0.0},
        }
    }
    with patch("torch.load", return_value=payload), patch("torch.save") as save_mock:
        ok, msg = evaluate_checkpoint(path, soft_min=0.0, settings=settings)
    assert ok is False
    assert "sem evidencia OOS" in msg
    save_mock.assert_not_called()
    assert payload["deploy_ok"] is False


def test_evaluate_checkpoint_rejects_missing_horizon(tmp_path: Path):
    path = tmp_path / "R_10.pth"
    path.write_bytes(b"x")
    settings = {
        "deep_learning": {
            "lookback": 480,
            "label_horizon_bars": 3,
            "train_timeframe": "micro",
            "deploy_gate": dict(_STRICT_GATE),
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


def test_evaluate_checkpoint_with_meta_path_qualifies(tmp_path: Path):
    path = tmp_path / "1HZ75V.pth"
    path.write_bytes(b"x")
    meta_path = tmp_path / "meta_lgbm.pkl"
    meta_path.write_bytes(b"meta_bytes")

    settings = {"deep_learning": {"deploy_gate": dict(_STRICT_GATE)}}
    meta_bundle = {
        "deploy_qualified": True,
        "oos_information_ratio": 1.57,
        "oos_payoff_zscore_mean": 0.045,
    }
    with (
        patch("torch.load", return_value={"val_accuracy": 0.55, "deploy_ok": False, **_COLLAPSE_OK}),
        patch("joblib.load", return_value=meta_bundle),
    ):
        ok, msg = evaluate_checkpoint(path, soft_min=0.50, settings=settings, meta_path=meta_path)
    assert ok is True
    assert "Two-Stage Stacking qualificado" in msg


def test_evaluate_checkpoint_with_meta_path_load_error(tmp_path: Path):
    path = tmp_path / "1HZ75V.pth"
    path.write_bytes(b"x")
    meta_path = tmp_path / "corrupt_meta.pkl"
    meta_path.write_bytes(b"bad")

    settings = {"deep_learning": {"deploy_gate": dict(_STRICT_GATE)}}
    with (
        patch("torch.load", return_value={"val_accuracy": 0.55, "deploy_ok": False, **_COLLAPSE_OK}),
        patch("joblib.load", side_effect=RuntimeError("load fail")),
    ):
        ok, msg = evaluate_checkpoint(path, soft_min=0.50, settings=settings, meta_path=meta_path)
    assert ok is False


def test_main_with_meta_flag(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("sys.argv", ["check_dl_deploy_gate.py", "--symbols", "1HZ75V", "--with-meta"])
    ckpt = tmp_path / "1HZ75V.pth"
    ckpt.write_bytes(b"x")
    with (
        patch("scripts.operations.check_dl_deploy_gate._checkpoint_paths", return_value=[ckpt]),
        patch("scripts.operations.check_dl_deploy_gate._load_settings", return_value={}),
        patch("scripts.operations.check_dl_deploy_gate._soft_min_acc", return_value=0.50),
        patch("scripts.operations.check_dl_deploy_gate.evaluate_checkpoint", return_value=(True, "ok")),
    ):
        from scripts.operations.check_dl_deploy_gate import main

        assert main() == 0
