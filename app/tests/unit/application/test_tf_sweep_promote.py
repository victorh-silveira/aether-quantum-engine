"""Testes de patch/promote/stamp do sweep multi-TF."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.application.services.deep_learning.tf_sweep_config import (
    load_tf_sweep_knobs,
    load_tf_sweep_manifest,
    resolve_enabled_candidates,
)
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
    assert patched["deep_learning"]["lookback"] == 288
    assert patched["deep_learning"]["label_ma_window"] == 3
    assert patched["deep_learning"]["indicators"]["windows"]["rsi_period"] == 6
    assert patched["orchestrator"]["execution"]["scale_vision"]["slope_bars"] == 2
    assert settings["data_handler"]["micro_granularity"] == 120
    train = patch_settings_for_sweep_train(settings, cand, artifact_root="data/dl/sweep")
    assert train["deep_learning"]["model_path_template"] == "data/dl/sweep/M5/{symbol}.pth"
    assert train["deep_learning"]["train_deploy_retries"] == 1
    assert train["infra"]["enabled"] is False
    train_infra = patch_settings_for_sweep_train(
        {**settings, "infra": {"enabled": True}},
        cand,
        artifact_root="data/dl/sweep",
        train_deploy_retries=2,
        disable_infra=False,
    )
    assert train_infra["deep_learning"]["train_deploy_retries"] == 2
    assert train_infra["infra"]["enabled"] is True


def test_promote_fail_closed_and_copy(tmp_path: Path):
    import torch

    knobs = load_tf_sweep_knobs(
        {
            "deep_learning": {"tf_sweep": {"min_edge_vs_breakeven": 0.03}},
            "risk_management": {"params": {"payout_estimate": 0.72}},
        }
    )
    artifact_root = "sweep_art"
    tf_dir = tmp_path / artifact_root / "M5"
    tf_dir.mkdir(parents=True)
    torch.save({"deploy_ok": False, "val_accuracy": 0.65, "val_brier": 0.22}, tf_dir / "R_10.pth")
    (tf_dir / "R_10_ts.pt").write_bytes(b"ts")
    dest = tmp_path / "live_dl"
    settings = {
        "data_handler": {"micro_granularity": 120, "mini_granularity": 120, "granularity": 3600},
        "deep_learning": {"lookback": 720, "model_path_template": "data/dl/{symbol}.pth"},
        "risk_management": {"params": {"duration": 2, "duration_unit": "m"}, "kelly": {}},
        "orchestrator": {
            "cycle_interval_seconds": 120,
            "signature_boundary_seconds": 120,
            "exec_empty_retry_seconds": 120,
        },
    }
    none_rows = [
        enrich_leaderboard_row(
            {"tf": "M2", "deploy_ok": True, "val_accuracy": 0.55, "val_brier": 0.25},
            knobs=knobs,
        )
    ]
    winner, untouched, copied = promote_winner_from_leaderboard(
        none_rows,
        settings,
        artifact_root=artifact_root,
        repo_root=tmp_path,
        copy_artifacts=True,
    )
    assert winner is None
    assert untouched["data_handler"]["micro_granularity"] == 120
    assert copied == []

    good = enrich_leaderboard_row(
        {
            "tf": "M5",
            "deploy_ok": False,
            "val_accuracy": 0.54,
            "settle_wr": 0.65,
            "settle_n": 24,
            "val_brier": 0.22,
            "granularity": 300,
            "micro_granularity": 300,
            "macro_granularity": 14400,
            "mini_granularity": 300,
            "duration": 5,
            "duration_unit": "m",
            "lookback": 288,
            "history_bars": 2000,
            "label_horizon_bars": 1,
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
    )
    assert winner is not None and winner["tf"] == "M5"
    assert patched["data_handler"]["micro_granularity"] == 300
    assert patched["risk_management"]["params"]["duration"] == 5
    stamped = torch.load(dest / "R_10.pth", map_location="cpu", weights_only=True)
    assert stamped["deploy_ok"] is True
    assert len(copied) == 2


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

    knobs = load_tf_sweep_knobs({"deep_learning": {"tf_sweep": {"payout_for_breakeven": 0.80}}, "risk_management": {}})
    assert knobs["payout_for_breakeven"] == pytest.approx(0.80)
    knobs2 = load_tf_sweep_knobs({"deep_learning": {}, "risk_management": {}})
    assert knobs2["payout_for_breakeven"] == pytest.approx(0.72)
    assert knobs2["train_deploy_retries"] == 1
    assert knobs2["disable_infra_during_sweep"] is True
    assert knobs2["min_settle_n"] == 16
    assert knobs2["min_history_bars"] == 800
    assert "launch_only" not in knobs2
    assert candidate_model_template("data/dl/sweep", "m5") == "data/dl/sweep/M5/{symbol}.pth"
    abs_p = resolve_repo_path(str(tmp_path / "x"), repo_root=tmp_path)
    assert abs_p == tmp_path / "x"
    with pytest.raises(FileNotFoundError):
        load_tf_sweep_manifest(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        load_tf_sweep_manifest(bad)
    bad.write_text('{"version":1,"defaults":[],"candidates":[]}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_tf_sweep_manifest(bad)
    bad.write_text('{"version":1,"defaults":{},"candidates":{}}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_tf_sweep_manifest(bad)
    rows = resolve_enabled_candidates(
        {
            "version": 1,
            "defaults": {"duration_unit": "m", "lookback": 720, "history_bars": 100, "label_horizon_bars": 1},
            "candidates": [
                "skip",
                {
                    "tf": "XX",
                    "enabled": False,
                    "micro_granularity": 120,
                    "macro_granularity": 3600,
                    "duration": 2,
                },
                {
                    "tf": "M2",
                    "enabled": True,
                    "micro_granularity": 120,
                    "macro_granularity": 3600,
                    "duration": 2,
                },
            ],
        }
    )
    assert [r["tf"] for r in rows] == ["M2"]
    with pytest.raises(ValueError):
        resolve_enabled_candidates(
            {
                "version": 1,
                "defaults": {},
                "candidates": [
                    {
                        "tf": "BAD",
                        "enabled": True,
                        "micro_granularity": 61,
                        "macro_granularity": 3600,
                        "duration": 1,
                    }
                ],
            }
        )
    with pytest.raises(ValueError):
        resolve_enabled_candidates(
            {
                "version": 1,
                "defaults": {},
                "candidates": [
                    {
                        "tf": "BAD",
                        "enabled": True,
                        "micro_granularity": 60,
                        "macro_granularity": 61,
                        "duration": 1,
                    }
                ],
            }
        )
