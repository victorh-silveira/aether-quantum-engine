"""Testes do sweep de horizonte N barras (R_10 M3)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.application.services.deep_learning.horizon_sweep import (
    DEFAULT_N_BARS,
    build_horizon_candidates,
    duration_minutes_for_n,
    load_horizon_sweep_knobs,
    parse_n_bars,
)
from src.application.services.deep_learning.tf_sweep_score import enrich_leaderboard_row, pick_tf_winner


def test_duration_minutes_matches_m3_grid():
    assert duration_minutes_for_n(1) == 3
    assert duration_minutes_for_n(2) == 6
    assert duration_minutes_for_n(3) == 9
    assert duration_minutes_for_n(5) == 15
    assert duration_minutes_for_n(5, micro_seconds=180) == 15


def test_parse_n_bars_default_and_dedupe():
    assert parse_n_bars(None) == DEFAULT_N_BARS
    assert parse_n_bars([]) == DEFAULT_N_BARS
    assert parse_n_bars([1, 2, 2, 3, 5]) == (1, 2, 3, 5)


def test_build_horizon_candidates_defaults_without_settings():
    rows = build_horizon_candidates({})
    assert [r["tf"] for r in rows] == ["H1", "H2", "H3", "H5"]
    assert rows[0]["lookback"] == 480
    override = build_horizon_candidates({}, n_bars=[1, 5])
    assert [r["tf"] for r in override] == ["H1", "H5"]
    assert override[1]["duration"] == 15
    settings = {
        "data_handler": {"micro_granularity": 180, "mini_granularity": 180, "granularity": 7200},
        "deep_learning": {"lookback": 480, "training_history_bars": 1333, "horizon_sweep": {"n_bars": [1, 2, 3, 5]}},
    }
    rows = build_horizon_candidates(settings)
    assert [r["tf"] for r in rows] == ["H1", "H2", "H3", "H5"]
    assert [r["label_horizon_bars"] for r in rows] == [1, 2, 3, 5]
    assert [r["duration"] for r in rows] == [3, 6, 9, 15]
    for row in rows:
        assert row["micro_granularity"] == 180
        assert row["macro_granularity"] == 7200
        assert row["lookback"] == 480
        assert row["history_bars"] == 1333
        assert row["duration_unit"] == "m"


def test_load_horizon_sweep_knobs_from_ssot():
    knobs = load_horizon_sweep_knobs()
    assert knobs["n_bars"] == [1, 2, 3, 5]
    assert knobs["run_in_launch_train"] is True
    assert knobs["enabled"] is True
    knobs = load_horizon_sweep_knobs({"deep_learning": {}, "risk_management": {"params": {"payout_estimate": 0.72}}})
    assert knobs["enabled"] is True
    assert knobs["run_in_launch_train"] is True
    assert knobs["n_bars"] == [1, 2, 3, 5]
    assert knobs["min_edge_vs_breakeven"] == pytest.approx(0.03)
    assert knobs["min_settle_n"] == 16
    assert knobs["min_history_bars"] == 800
    assert knobs["auto_promote"] is True
    custom = load_horizon_sweep_knobs(
        {
            "deep_learning": {
                "horizon_sweep": {
                    "n_bars": [1, 5],
                    "payout_for_breakeven": 0.80,
                    "symbols": ["R_10"],
                    "enabled": True,
                    "run_in_launch_train": False,
                }
            },
            "risk_management": {},
        }
    )
    assert custom["n_bars"] == [1, 5]
    assert custom["payout_for_breakeven"] == pytest.approx(0.80)
    assert custom["run_in_launch_train"] is False
    assert custom["symbols"] == ["R_10"]


def test_pick_horizon_winner_among_eligible():
    knobs = load_horizon_sweep_knobs({"deep_learning": {}, "risk_management": {"params": {"payout_estimate": 0.72}}})
    rows = [
        enrich_leaderboard_row(
            {
                "tf": "H1",
                "label_horizon_bars": 1,
                "duration": 3,
                "deploy_ok": True,
                "val_accuracy": 0.60,
                "settle_wr": 0.62,
                "settle_n": 24,
                "history_bars": 1333,
                "val_brier": 0.22,
            },
            knobs=knobs,
        ),
        enrich_leaderboard_row(
            {
                "tf": "H3",
                "label_horizon_bars": 3,
                "duration": 9,
                "deploy_ok": True,
                "val_accuracy": 0.54,
                "settle_wr": 0.70,
                "settle_n": 24,
                "history_bars": 1333,
                "val_brier": 0.21,
            },
            knobs=knobs,
        ),
        enrich_leaderboard_row(
            {
                "tf": "H5",
                "label_horizon_bars": 5,
                "duration": 15,
                "deploy_ok": True,
                "val_accuracy": 0.58,
                "settle_wr": 0.50,
                "settle_n": 24,
                "history_bars": 1333,
                "val_brier": 0.20,
            },
            knobs=knobs,
        ),
    ]
    winner = pick_tf_winner(rows)
    assert winner is not None
    assert winner["tf"] == "H3"
    assert int(winner["label_horizon_bars"]) == 3


def test_launch_train_pipeline_runs_horizon():
    from scripts.operations.run_launch_train_tf_pipeline import run_launch_train_tf_pipeline

    settings = {
        "data_handler": {"micro_granularity": 180, "mini_granularity": 180, "granularity": 7200},
        "deep_learning": {
            "lookback": 480,
            "training_history_bars": 1333,
            "horizon_sweep": {"enabled": True, "run_in_launch_train": True, "n_bars": [1, 2, 3, 5]},
        },
        "risk_management": {"params": {"payout_estimate": 0.72}},
    }
    captured: dict = {}

    def _fake_sweep(**kwargs):
        captured["candidates"] = kwargs.get("candidates")
        captured["knobs"] = kwargs.get("knobs")
        return []

    with (
        patch("scripts.operations.run_launch_train_tf_pipeline.load_settings_json", return_value=settings),
        patch("scripts.operations.run_launch_train_tf_pipeline.clear_sweep_artifacts", return_value=None),
        patch("scripts.operations.run_launch_train_tf_pipeline.run_tf_sweep", side_effect=_fake_sweep),
        patch("scripts.operations.run_launch_train_tf_pipeline.subprocess.run") as sp,
    ):
        assert run_launch_train_tf_pipeline() == 1
    sp.assert_not_called()
    cands = captured.get("candidates") or []
    assert [c["tf"] for c in cands] == ["H1", "H2", "H3", "H5"]
    assert captured["knobs"]["run_in_launch_train"] is True
