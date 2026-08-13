"""Testes CLI do promote/sweep TF."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from scripts.operations.promote_tf_winner import main as promote_main
from scripts.operations.sweep_train_timeframes import (
    main as sweep_main,
    run_tf_sweep,
)
from src.application.services.deep_learning.tf_sweep_config import load_tf_sweep_knobs
from src.application.services.deep_learning.tf_sweep_score import enrich_leaderboard_row


def test_sweep_main_dry_run(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = {
        "data_handler": {},
        "deep_learning": {
            "tf_sweep": {
                "enabled": True,
                "artifact_root": "art",
                "leaderboard_path": "art/lb.json",
                "min_edge_vs_breakeven": 0.03,
            }
        },
        "risk_management": {"params": {"payout_estimate": 0.72}},
    }
    with (
        patch("scripts.operations.sweep_train_timeframes.load_settings_json", return_value=settings),
        patch(
            "scripts.operations.sweep_train_timeframes.resolve_enabled_candidates",
            return_value=[
                {
                    "tf": "M2",
                    "micro_granularity": 120,
                    "macro_granularity": 3600,
                    "mini_granularity": 120,
                    "duration": 2,
                    "duration_unit": "m",
                    "lookback": 720,
                    "history_bars": 2000,
                    "label_horizon_bars": 1,
                    "train_timeframe": "micro",
                }
            ],
        ),
        patch("scripts.operations.sweep_train_timeframes.REPO_ROOT", tmp_path),
        patch(
            "scripts.operations.sweep_train_timeframes.load_tf_sweep_knobs",
            return_value=load_tf_sweep_knobs(settings),
        ),
    ):
        assert sweep_main(["--dry-run", "--only", "M2"]) == 0
    assert (tmp_path / "art" / "lb.json").is_file()


def test_promote_main_fail_closed(tmp_path: Path):
    knobs = {
        "payout_for_breakeven": 0.72,
        "min_edge_vs_breakeven": 0.03,
        "weight_edge": 1.0,
        "weight_brier": 0.5,
        "weight_sharpness": 0.25,
        "weight_meta_ir": 0.25,
        "soft_max_brier": 0.26,
        "artifact_root": "art",
        "leaderboard_path": str(tmp_path / "lb.json"),
        "enabled": True,
    }
    row = enrich_leaderboard_row(
        {"tf": "M2", "deploy_ok": True, "val_accuracy": 0.55, "val_brier": 0.24},
        knobs=knobs,
    )
    (tmp_path / "lb.json").write_text(json.dumps({"version": 1, "rows": [row]}), encoding="utf-8")
    with (
        patch("scripts.operations.promote_tf_winner.load_settings_json", return_value={}),
        patch("scripts.operations.promote_tf_winner.load_tf_sweep_knobs", return_value=knobs),
    ):
        assert promote_main(["--leaderboard", str(tmp_path / "lb.json")]) == 1


def test_promote_main_dry_run_ok(tmp_path: Path):
    knobs = {
        "payout_for_breakeven": 0.72,
        "min_edge_vs_breakeven": 0.03,
        "weight_edge": 1.0,
        "weight_brier": 0.5,
        "weight_sharpness": 0.25,
        "weight_meta_ir": 0.25,
        "soft_max_brier": 0.26,
        "artifact_root": "art",
        "leaderboard_path": str(tmp_path / "lb.json"),
        "enabled": True,
    }
    row = enrich_leaderboard_row(
        {
            "tf": "M5",
            "deploy_ok": True,
            "val_accuracy": 0.54,
            "settle_wr": 0.65,
            "settle_n": 24,
            "history_bars": 2000,
            "val_brier": 0.22,
            "granularity": 300,
            "macro_granularity": 14400,
            "duration": 5,
        },
        knobs=knobs,
    )
    (tmp_path / "lb.json").write_text(json.dumps({"version": 1, "rows": [row]}), encoding="utf-8")
    with (
        patch("scripts.operations.promote_tf_winner.load_settings_json", return_value={}),
        patch("scripts.operations.promote_tf_winner.load_tf_sweep_knobs", return_value=knobs),
    ):
        assert promote_main(["--leaderboard", str(tmp_path / "lb.json"), "--dry-run"]) == 0


def test_launch_train_pipeline_dry_and_no_eligible(tmp_path: Path, monkeypatch):
    from scripts.operations.run_launch_train_tf_pipeline import (
        clear_sweep_artifacts,
        run_launch_train_tf_pipeline,
    )

    art = tmp_path / "art"
    art.mkdir()
    (art / "stale.txt").write_text("x", encoding="utf-8")
    clear_sweep_artifacts("art", repo_root=tmp_path)
    assert not (art / "stale.txt").exists()

    settings = {
        "deep_learning": {
            "tf_sweep": {
                "enabled": True,
                "run_in_launch_train": True,
                "auto_promote": True,
                "min_edge_vs_breakeven": 0.03,
                "artifact_root": "art",
                "leaderboard_path": "art/lb.json",
            }
        },
        "risk_management": {"params": {"payout_estimate": 0.72}},
        "data_handler": {},
        "orchestrator": {
            "cycle_interval_seconds": 120,
            "signature_boundary_seconds": 120,
            "exec_empty_retry_seconds": 120,
        },
    }
    knobs = load_tf_sweep_knobs(settings)
    board = [
        enrich_leaderboard_row(
            {"tf": "M2", "deploy_ok": True, "val_accuracy": 0.55, "val_brier": 0.24},
            knobs=knobs,
        )
    ]
    with (
        patch(
            "scripts.operations.run_launch_train_tf_pipeline.load_settings_json",
            return_value=settings,
        ),
        patch(
            "scripts.operations.run_launch_train_tf_pipeline.load_tf_sweep_knobs",
            return_value=knobs,
        ),
        patch(
            "scripts.operations.run_launch_train_tf_pipeline.run_tf_sweep",
            return_value=board,
        ),
        patch("scripts.operations.run_launch_train_tf_pipeline.REPO_ROOT", tmp_path),
        patch(
            "scripts.operations.run_launch_train_tf_pipeline.clear_sweep_artifacts",
            return_value=None,
        ),
    ):
        assert run_launch_train_tf_pipeline(dry_run=True) == 0
        assert run_launch_train_tf_pipeline() == 1


def test_launch_train_pipeline_promotes_winner(tmp_path: Path):
    from scripts.operations.run_launch_train_tf_pipeline import run_launch_train_tf_pipeline

    settings = {
        "deep_learning": {
            "tf_sweep": {
                "enabled": True,
                "run_in_launch_train": True,
                "auto_promote": True,
                "min_edge_vs_breakeven": 0.03,
                "artifact_root": "art",
                "leaderboard_path": "art/lb.json",
            }
        },
        "risk_management": {"params": {"payout_estimate": 0.72}},
    }
    knobs = load_tf_sweep_knobs(settings)
    board = [
        enrich_leaderboard_row(
            {
                "tf": "M5",
                "deploy_ok": True,
                "val_accuracy": 0.54,
                "settle_wr": 0.65,
                "settle_n": 24,
                "history_bars": 2000,
                "val_brier": 0.22,
                "granularity": 300,
                "macro_granularity": 14400,
                "duration": 5,
            },
            knobs=knobs,
        )
    ]
    with (
        patch(
            "scripts.operations.run_launch_train_tf_pipeline.load_settings_json",
            return_value=settings,
        ),
        patch(
            "scripts.operations.run_launch_train_tf_pipeline.load_tf_sweep_knobs",
            return_value=knobs,
        ),
        patch(
            "scripts.operations.run_launch_train_tf_pipeline.run_tf_sweep",
            return_value=board,
        ),
        patch(
            "scripts.operations.run_launch_train_tf_pipeline.clear_sweep_artifacts",
            return_value=None,
        ),
        patch(
            "scripts.operations.run_launch_train_tf_pipeline.promote_main",
            return_value=0,
        ) as promo,
    ):
        assert run_launch_train_tf_pipeline() == 0
        promo.assert_called_once()


def test_launch_train_pipeline_fallback_single(tmp_path: Path):
    from scripts.operations.run_launch_train_tf_pipeline import run_launch_train_tf_pipeline

    knobs = {
        "enabled": False,
        "run_in_launch_train": False,
        "min_edge_vs_breakeven": 0.03,
        "artifact_root": "art",
    }
    with (
        patch(
            "scripts.operations.run_launch_train_tf_pipeline.load_settings_json",
            return_value={},
        ),
        patch(
            "scripts.operations.run_launch_train_tf_pipeline.load_tf_sweep_knobs",
            return_value=knobs,
        ),
        patch("scripts.operations.run_launch_train_tf_pipeline.REPO_ROOT", tmp_path),
        patch(
            "scripts.operations.run_launch_train_tf_pipeline.subprocess.run",
            return_value=type("R", (), {"returncode": 0})(),
        ) as sp,
    ):
        assert run_launch_train_tf_pipeline() == 0
        sp.assert_called_once()


def test_run_tf_sweep_dry_writes_overlay(tmp_path: Path):
    settings = {
        "data_handler": {"micro_granularity": 120},
        "deep_learning": {"lookback": 720},
        "risk_management": {"params": {"duration": 2, "duration_unit": "m"}, "kelly": {}},
        "orchestrator": {
            "cycle_interval_seconds": 120,
            "signature_boundary_seconds": 120,
            "exec_empty_retry_seconds": 120,
        },
    }
    knobs = {
        "payout_for_breakeven": 0.72,
        "min_edge_vs_breakeven": 0.03,
        "weight_edge": 1.0,
        "weight_brier": 0.5,
        "weight_sharpness": 0.25,
        "weight_meta_ir": 0.25,
        "soft_max_brier": 0.26,
        "artifact_root": "art",
        "leaderboard_path": "art/lb.json",
    }
    cand = {
        "tf": "M3",
        "micro_granularity": 180,
        "macro_granularity": 7200,
        "mini_granularity": 180,
        "duration": 3,
        "duration_unit": "m",
        "lookback": 720,
        "history_bars": 2000,
        "label_horizon_bars": 1,
        "train_timeframe": "micro",
    }
    board = run_tf_sweep(
        settings=settings,
        knobs=knobs,
        candidates=[cand],
        dry_run=True,
        repo_root=tmp_path,
    )
    assert board[0]["error"] == "dry_run"
    overlay = tmp_path / "art" / "M3" / "settings_overlay.json"
    assert overlay.is_file()
    import json

    payload = json.loads(overlay.read_text(encoding="utf-8"))
    assert payload["infra"]["enabled"] is False
    assert payload["deep_learning"]["train_deploy_retries"] == 1
