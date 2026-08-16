"""Testes de patch/promote/stamp do sweep de horizonte."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.application.services.deep_learning.tf_sweep_config import load_tf_sweep_knobs
from src.application.services.deep_learning.tf_sweep_promote import (
    patch_settings_for_candidate,
    patch_settings_for_sweep_train,
    promote_winner_from_leaderboard,
)
from src.application.services.deep_learning.tf_sweep_score import enrich_leaderboard_row


def test_patch_settings_aligns_contract_and_cycle():
    settings = {
        "data_handler": {"micro_granularity": 120, "mini_granularity": 120, "granularity": 3600},
        "deep_learning": {
            "lookback": 720,
            "train_timeframe": "micro",
            "label_ma_window": 8,
            "label_smooth_bars": 2,
            "implied_vol_bars": 120,
            "indicators": {"windows": {"rsi_period": 14, "bb_window": 20}},
            "deploy_gate": {"mini_bars": 120},
        },
        "risk_management": {"params": {"duration": 2, "duration_unit": "m"}, "kelly": {}},
        "orchestrator": {
            "cycle_interval_seconds": 120,
            "signature_boundary_seconds": 120,
            "exec_empty_retry_seconds": 120,
            "execution": {
                "scale_vision": {"slope_bars": 5},
                "dynamic_threshold": {"baseline_lookback": 72},
            },
        },
    }
    cand = {
        "tf": "M5",
        "micro_granularity": 300,
        "macro_granularity": 14400,
        "mini_granularity": 300,
        "duration": 5,
        "duration_unit": "m",
        "label_horizon_bars": 1,
    }
    patched = patch_settings_for_candidate(settings, cand)
    assert patched["data_handler"]["micro_granularity"] == 300
    assert patched["data_handler"]["granularity"] == 14400
    assert patched["risk_management"]["params"]["duration"] == 5
    assert patched["orchestrator"]["cycle_interval_seconds"] == 300
    assert patched["deep_learning"]["lookback"] == 720
    assert patched["deep_learning"]["label_ma_window"] == 8
    assert patched["deep_learning"]["indicators"]["windows"]["rsi_period"] == 14
    assert patched["orchestrator"]["execution"]["scale_vision"]["slope_bars"] == 5
    assert settings["data_handler"]["micro_granularity"] == 120
    ops_fixed = patch_settings_for_candidate(
        settings,
        {**cand, "duration": 50, "label_horizon_bars": 50},
        ops_contract_duration_minutes=5,
    )
    assert ops_fixed["risk_management"]["params"]["duration"] == 5
    assert ops_fixed["deep_learning"]["label_horizon_bars"] == 50
    train = patch_settings_for_sweep_train(settings, cand, artifact_root="data/dl/sweep")
    assert train["risk_management"]["params"]["duration"] == 5
    assert train["deep_learning"]["model_path_template"] == "data/dl/sweep/R_10/M5/{symbol}.pth"
    assert train["logging"]["level"] == "CRITICAL"
    assert int(train["deep_learning"]["training_log_every_n_epochs"]) >= 10**9
    noisy = patch_settings_for_sweep_train(settings, cand, artifact_root="data/dl/sweep", quiet_train_logs=False)
    assert noisy.get("logging", {}).get("level") != "CRITICAL" or "level" not in noisy.get("logging", {})
    assert int(noisy["deep_learning"].get("training_log_every_n_epochs") or 0) < 10**9 or (
        "training_log_every_n_epochs" not in noisy["deep_learning"]
    )
    assert train["deep_learning"]["train_deploy_retries"] == 1
    assert train["infra"]["enabled"] is False
    assert train["anchor"] == "R_10"
    assert train["symbols"] == ["R_10"]
    assert train["deep_learning"]["train_symbols"] == ["R_10"]
    train_infra = patch_settings_for_sweep_train(
        {**settings, "infra": {"enabled": True}},
        cand,
        artifact_root="data/dl/sweep",
        symbol="R_25",
        train_deploy_retries=2,
        disable_infra=False,
    )
    assert train_infra["deep_learning"]["train_deploy_retries"] == 2
    assert train_infra["infra"]["enabled"] is True
    assert train_infra["deep_learning"]["model_path_template"] == "data/dl/sweep/R_25/M5/{symbol}.pth"
    assert train_infra["anchor"] == "R_25"


def test_promote_fail_closed_and_copy(tmp_path: Path):
    import torch

    knobs = load_tf_sweep_knobs(
        {
            "deep_learning": {"horizon_sweep": {"min_edge_vs_breakeven": 0.03}},
            "risk_management": {"params": {"payout_estimate": 0.72}},
        }
    )
    artifact_root = "sweep_art"
    tf_dir = tmp_path / artifact_root / "R_10" / "H50"
    tf_dir.mkdir(parents=True)
    torch.save({"deploy_ok": False, "val_accuracy": 0.65, "val_brier": 0.22}, tf_dir / "R_10.pth")
    (tf_dir / "R_10_ts.pt").write_bytes(b"ts")
    dest = tmp_path / "live_dl"
    drift_path = tmp_path / "drift_symbols.py"
    settings = {
        "anchor": "R_10",
        "symbols": ["R_10"],
        "data_handler": {"micro_granularity": 120, "mini_granularity": 120, "granularity": 3600},
        "deep_learning": {
            "lookback": 720,
            "model_path_template": "data/dl/{symbol}.pth",
            "horizon_sweep": {"ops_contract_duration_minutes": 5},
        },
        "risk_management": {"params": {"duration": 2, "duration_unit": "m"}, "kelly": {}},
        "orchestrator": {
            "cycle_interval_seconds": 120,
            "signature_boundary_seconds": 120,
            "exec_empty_retry_seconds": 120,
        },
    }
    none_rows = [
        enrich_leaderboard_row(
            {"symbol": "R_10", "tf": "M2", "deploy_ok": True, "val_accuracy": 0.55, "val_brier": 0.25},
            knobs=knobs,
        )
    ]
    winner, untouched, copied = promote_winner_from_leaderboard(
        none_rows,
        settings,
        artifact_root=artifact_root,
        repo_root=tmp_path,
        copy_artifacts=True,
        write_symbols_module=False,
    )
    assert winner is None
    assert untouched["data_handler"]["micro_granularity"] == 120
    assert copied == []

    good = enrich_leaderboard_row(
        {
            "symbol": "R_10",
            "tf": "H50",
            "deploy_ok": False,
            "val_accuracy": 0.54,
            "settle_wr": 0.65,
            "settle_n": 24,
            "val_brier": 0.22,
            "granularity": 60,
            "micro_granularity": 60,
            "macro_granularity": 7200,
            "mini_granularity": 60,
            "duration": 50,
            "duration_unit": "m",
            "lookback": 480,
            "history_bars": 1333,
            "label_horizon_bars": 50,
        },
        knobs=knobs,
    )
    winner, patched, copied = promote_winner_from_leaderboard(
        [good],
        settings,
        artifact_root=artifact_root,
        dest_dir=dest,
        repo_root=tmp_path,
        copy_artifacts=True,
        write_symbols_module=True,
        symbols_module_path=drift_path,
    )
    assert winner is not None and winner["tf"] == "H50"
    assert patched["data_handler"]["micro_granularity"] == 60
    assert patched["risk_management"]["params"]["duration"] == 5
    assert patched["deep_learning"]["label_horizon_bars"] == 50
    assert patched["anchor"] == "R_10"
    assert patched["symbols"] == ["R_10"]
    stamped = torch.load(dest / "R_10.pth", map_location="cpu", weights_only=True)
    assert stamped["deploy_ok"] is True
    assert len(copied) == 2
    assert 'TRADING_SYMBOLS: tuple[str, ...] = ("R_10",)' in drift_path.read_text(encoding="utf-8")


def test_promote_rejects_corrupt_settings_blocks():
    cand = {
        "tf": "M5",
        "micro_granularity": 300,
        "macro_granularity": 14400,
        "duration": 5,
        "duration_unit": "m",
    }
    with pytest.raises(ValueError):
        patch_settings_for_candidate({"data_handler": []}, cand)
    with pytest.raises(ValueError):
        patch_settings_for_candidate({"data_handler": {}, "deep_learning": []}, cand)
    with pytest.raises(ValueError):
        patch_settings_for_candidate({"data_handler": {}, "deep_learning": {}, "risk_management": []}, cand)
    with pytest.raises(ValueError):
        patch_settings_for_candidate({"data_handler": {}, "deep_learning": {}, "risk_management": {"params": []}}, cand)
    with pytest.raises(ValueError):
        patch_settings_for_candidate(
            {
                "data_handler": {},
                "deep_learning": {},
                "risk_management": {"params": {}},
                "orchestrator": [],
            },
            cand,
        )


def test_promote_artifacts_missing_raises(tmp_path: Path):
    from src.application.services.deep_learning.tf_sweep_promote import promote_artifacts

    with pytest.raises(FileNotFoundError):
        promote_artifacts(artifact_root="art", tf="M5", repo_root=tmp_path)


def test_stamp_checkpoint_deploy_ok_invalid_and_already_ok(tmp_path: Path):
    import torch

    from src.application.services.deep_learning.tf_sweep_promote import _stamp_checkpoint_deploy_ok

    bad = tmp_path / "bad.pth"
    torch.save([1, 2, 3], bad)
    with pytest.raises(ValueError, match="checkpoint invalido"):
        _stamp_checkpoint_deploy_ok(bad)

    ok = tmp_path / "ok.pth"
    torch.save({"deploy_ok": True, "val_accuracy": 0.6}, ok)
    _stamp_checkpoint_deploy_ok(ok)
    payload = torch.load(ok, map_location="cpu", weights_only=True)
    assert payload["deploy_ok"] is True


def test_config_edge_paths_and_template(tmp_path: Path):
    from src.application.services.deep_learning.tf_sweep_config import (
        candidate_model_template,
        resolve_repo_path,
    )

    knobs = load_tf_sweep_knobs(
        {"deep_learning": {"horizon_sweep": {"payout_for_breakeven": 0.80}}, "risk_management": {}}
    )
    assert knobs["payout_for_breakeven"] == pytest.approx(0.80)
    knobs2 = load_tf_sweep_knobs({"deep_learning": {}, "risk_management": {}})
    assert knobs2["payout_for_breakeven"] == pytest.approx(0.72)
    assert knobs2["train_deploy_retries"] == 1
    assert knobs2["disable_infra_during_sweep"] is True
    assert knobs2["min_settle_n"] == 16
    assert knobs2["min_history_bars"] == 800
    assert knobs2["symbols"] == ["R_10"]
    assert knobs2["n_bars"] == [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    assert "launch_only" not in knobs2
    assert candidate_model_template("data/dl/sweep", "h5") == "data/dl/sweep/R_10/H5/{symbol}.pth"
    assert candidate_model_template("data/dl/sweep", "h5", symbol="R_75") == "data/dl/sweep/R_75/H5/{symbol}.pth"
    abs_p = resolve_repo_path(str(tmp_path / "x"), repo_root=tmp_path)
    assert abs_p == tmp_path / "x"
