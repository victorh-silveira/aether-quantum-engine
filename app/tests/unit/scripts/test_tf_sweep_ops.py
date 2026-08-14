"""Testes CLI do promote/sweep de horizonte."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from scripts.operations.promote_tf_winner import main as promote_main
from scripts.operations.sweep_train_timeframes import (
    main as sweep_main,
    run_tf_sweep,
)
from src.application.services.deep_learning.horizon_sweep import load_horizon_sweep_knobs
from src.application.services.deep_learning.tf_sweep_score import enrich_leaderboard_row


def test_sweep_main_dry_run(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = {
        "data_handler": {},
        "deep_learning": {
            "horizon_sweep": {
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
        patch("scripts.operations.sweep_train_timeframes.REPO_ROOT", tmp_path),
    ):
        assert sweep_main(["--dry-run", "--only", "H1"]) == 0
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
        {"tf": "H1", "deploy_ok": True, "val_accuracy": 0.55, "val_brier": 0.24},
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
            "tf": "H5",
            "deploy_ok": True,
            "val_accuracy": 0.54,
            "settle_wr": 0.65,
            "settle_n": 24,
            "history_bars": 2000,
            "val_brier": 0.22,
            "granularity": 180,
            "macro_granularity": 7200,
            "duration": 15,
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
            "horizon_sweep": {
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
            "cycle_interval_seconds": 180,
            "signature_boundary_seconds": 180,
            "exec_empty_retry_seconds": 180,
        },
    }
    knobs = load_horizon_sweep_knobs(settings)
    board = [
        enrich_leaderboard_row(
            {"tf": "H1", "deploy_ok": True, "val_accuracy": 0.55, "val_brier": 0.24},
            knobs=knobs,
        )
    ]
    with (
        patch(
            "scripts.operations.run_launch_train_tf_pipeline.load_settings_json",
            return_value=settings,
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
            "horizon_sweep": {
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
    knobs = load_horizon_sweep_knobs(settings)
    board = [
        enrich_leaderboard_row(
            {
                "tf": "H5",
                "deploy_ok": True,
                "val_accuracy": 0.54,
                "settle_wr": 0.65,
                "settle_n": 24,
                "history_bars": 2000,
                "val_brier": 0.22,
                "granularity": 180,
                "macro_granularity": 7200,
                "duration": 15,
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
        promo.reset_mock()
        assert run_launch_train_tf_pipeline(skip_promote=True) == 0
        promo.assert_not_called()


def test_launch_train_pipeline_fallback_single(tmp_path: Path):
    from scripts.operations.run_launch_train_tf_pipeline import run_launch_train_tf_pipeline

    with (
        patch(
            "scripts.operations.run_launch_train_tf_pipeline.load_settings_json",
            return_value={},
        ),
        patch(
            "scripts.operations.run_launch_train_tf_pipeline.load_horizon_sweep_knobs",
            return_value={"enabled": False, "run_in_launch_train": False},
        ),
        patch("scripts.operations.run_launch_train_tf_pipeline.REPO_ROOT", tmp_path),
        patch(
            "scripts.operations.run_launch_train_tf_pipeline.subprocess.run",
            return_value=type("R", (), {"returncode": 0})(),
        ) as sp,
    ):
        assert run_launch_train_tf_pipeline() == 0
        sp.assert_called_once()
        assert run_launch_train_tf_pipeline(dry_run=True) == 0


def test_run_tf_sweep_dry_writes_overlay(tmp_path: Path):
    settings = {
        "data_handler": {"micro_granularity": 180},
        "deep_learning": {"lookback": 480},
        "risk_management": {"params": {"duration": 9, "duration_unit": "m"}, "kelly": {}},
        "orchestrator": {
            "cycle_interval_seconds": 180,
            "signature_boundary_seconds": 180,
            "exec_empty_retry_seconds": 180,
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
        "symbols": ["R_10"],
    }
    cand = {
        "tf": "H3",
        "micro_granularity": 180,
        "macro_granularity": 7200,
        "mini_granularity": 180,
        "duration": 9,
        "duration_unit": "m",
        "lookback": 480,
        "history_bars": 1333,
        "label_horizon_bars": 3,
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
    assert board[0]["symbol"] == "R_10"
    overlay = tmp_path / "art" / "R_10" / "H3" / "settings_overlay.json"
    assert overlay.is_file()
    payload = json.loads(overlay.read_text(encoding="utf-8"))
    assert payload["infra"]["enabled"] is False
    assert payload["deep_learning"]["train_deploy_retries"] == 1
    assert payload["anchor"] == "R_10"
    assert payload["symbols"] == ["R_10"]
    assert payload["deep_learning"]["lookback"] == 480
    assert payload["deep_learning"]["label_horizon_bars"] == 3


def test_run_tf_sweep_with_mock_train(tmp_path: Path):
    settings = {
        "data_handler": {"micro_granularity": 180},
        "deep_learning": {"lookback": 480},
        "risk_management": {"params": {"duration": 9, "duration_unit": "m"}, "kelly": {}},
        "orchestrator": {
            "cycle_interval_seconds": 180,
            "signature_boundary_seconds": 180,
            "exec_empty_retry_seconds": 180,
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
        "symbols": ["R_10"],
        "min_settle_n": 16,
        "min_history_bars": 800,
    }
    cand = {
        "tf": "H3",
        "micro_granularity": 180,
        "macro_granularity": 7200,
        "mini_granularity": 180,
        "duration": 9,
        "duration_unit": "m",
        "lookback": 480,
        "history_bars": 1333,
        "label_horizon_bars": 3,
        "train_timeframe": "micro",
    }

    def _train(_patched, _candidate, _art, symbol):
        return {
            "deploy_ok": True,
            "val_accuracy": 0.54,
            "val_brier": 0.22,
            "settle_wr": 0.65,
            "settle_n": 24,
            "history_bars": 1333,
            "symbol": symbol,
        }

    board = run_tf_sweep(
        settings=settings,
        knobs=knobs,
        candidates=[cand],
        train_fn=_train,
        repo_root=tmp_path,
    )
    assert board[0]["eligible"] is True
    assert board[0]["tf"] == "H3"
    assert (tmp_path / "art" / "lb.json").is_file()


def test_run_tf_sweep_multi_symbol_dry(tmp_path: Path):
    settings = {
        "data_handler": {"micro_granularity": 180},
        "deep_learning": {"lookback": 480},
        "risk_management": {"params": {"duration": 9, "duration_unit": "m"}, "kelly": {}},
        "orchestrator": {
            "cycle_interval_seconds": 180,
            "signature_boundary_seconds": 180,
            "exec_empty_retry_seconds": 180,
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
        "symbols": ["R_10", "R_25"],
    }
    cand = {
        "tf": "H1",
        "micro_granularity": 180,
        "macro_granularity": 7200,
        "mini_granularity": 180,
        "duration": 3,
        "duration_unit": "m",
        "lookback": 480,
        "history_bars": 1333,
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
    assert len(board) == 2
    assert {row["symbol"] for row in board} == {"R_10", "R_25"}
    assert (tmp_path / "art" / "R_25" / "H1" / "settings_overlay.json").is_file()
