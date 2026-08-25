"""Testes de score e knobs do sweep de horizonte."""

from __future__ import annotations

import pytest

from src.application.services.deep_learning.tf_sweep_config import load_tf_sweep_knobs
from src.application.services.deep_learning.tf_sweep_score import (
    enrich_leaderboard_row,
    implied_breakeven,
    is_tf_eligible,
    pick_tf_winner,
    score_tf_row,
)


def test_implied_breakeven_matches_live_logs():
    assert implied_breakeven(0.72) == pytest.approx(1.0 / 1.72, rel=1e-6)


def test_eligibility_requires_edge_above_breakeven():
    be = implied_breakeven(0.72)
    assert (
        is_tf_eligible(
            rank_wr=0.54,
            be_implied=be,
            min_edge_vs_breakeven=0.03,
            settle_n=24,
            min_settle_n=16,
        )
        is False
    )
    assert (
        is_tf_eligible(
            rank_wr=be + 0.03,
            be_implied=be,
            min_edge_vs_breakeven=0.03,
            settle_n=24,
            min_settle_n=16,
        )
        is True
    )
    assert (
        is_tf_eligible(
            rank_wr=0.70,
            be_implied=be,
            min_edge_vs_breakeven=0.03,
            settle_n=8,
            min_settle_n=16,
        )
        is False
    )


def test_pick_winner_fail_closed_when_none_eligible():
    knobs = {
        "payout_for_breakeven": 0.72,
        "min_edge_vs_breakeven": 0.03,
        "min_settle_n": 16,
        "min_history_bars": 800,
        "weight_edge": 1.0,
        "weight_brier": 0.5,
        "weight_sharpness": 0.25,
        "weight_meta_ir": 0.25,
        "soft_max_brier": 0.26,
    }
    rows = [
        enrich_leaderboard_row(
            {
                "tf": "H1",
                "deploy_ok": False,
                "val_accuracy": 0.70,
                "settle_wr": 0.55,
                "settle_n": 24,
                "history_bars": 2000,
                "val_brier": 0.24,
            },
            knobs=knobs,
        ),
        enrich_leaderboard_row(
            {
                "tf": "H5",
                "deploy_ok": False,
                "val_accuracy": 0.70,
                "settle_wr": 0.56,
                "settle_n": 24,
                "history_bars": 2000,
                "val_brier": 0.23,
            },
            knobs=knobs,
        ),
    ]
    assert all(r["eligible"] is False for r in rows)
    assert pick_tf_winner(rows) is None


def test_pick_winner_uses_settle_wr_not_label_acc():
    knobs = {
        "payout_for_breakeven": 0.72,
        "min_edge_vs_breakeven": 0.03,
        "min_settle_n": 16,
        "min_history_bars": 800,
        "weight_edge": 1.0,
        "weight_brier": 0.5,
        "weight_sharpness": 0.0,
        "weight_meta_ir": 0.0,
        "soft_max_brier": 0.26,
    }
    high_label = enrich_leaderboard_row(
        {
            "tf": "H1",
            "deploy_ok": False,
            "val_accuracy": 0.80,
            "settle_wr": 0.55,
            "settle_n": 24,
            "history_bars": 2000,
            "val_brier": 0.24,
        },
        knobs=knobs,
    )
    high_settle = enrich_leaderboard_row(
        {
            "tf": "H5",
            "deploy_ok": False,
            "val_accuracy": 0.54,
            "settle_wr": 0.65,
            "settle_n": 24,
            "history_bars": 2000,
            "val_brier": 0.22,
        },
        knobs=knobs,
    )
    thin = enrich_leaderboard_row(
        {
            "tf": "H3",
            "deploy_ok": True,
            "val_accuracy": 0.58,
            "settle_wr": 0.83,
            "settle_n": 8,
            "history_bars": 496,
            "val_brier": 0.22,
        },
        knobs=knobs,
    )
    assert high_label["eligible"] is False
    assert high_settle["eligible"] is True
    assert thin["eligible"] is False
    assert pick_tf_winner([high_label, high_settle, thin])["tf"] == "H5"


def test_pick_winner_argmax_score_with_tiebreak():
    knobs = {
        "payout_for_breakeven": 0.72,
        "min_edge_vs_breakeven": 0.03,
        "min_settle_n": 16,
        "min_history_bars": 0,
        "weight_edge": 1.0,
        "weight_brier": 0.5,
        "weight_sharpness": 0.0,
        "weight_meta_ir": 0.0,
        "soft_max_brier": 0.26,
    }
    weak = enrich_leaderboard_row(
        {
            "tf": "H1",
            "deploy_ok": True,
            "settle_wr": 0.62,
            "settle_n": 24,
            "val_accuracy": 0.62,
            "val_brier": 0.24,
            "oos_sharpness": 0.01,
        },
        knobs=knobs,
    )
    strong = enrich_leaderboard_row(
        {
            "tf": "H5",
            "deploy_ok": True,
            "settle_wr": 0.65,
            "settle_n": 24,
            "val_accuracy": 0.65,
            "val_brier": 0.22,
            "oos_sharpness": 0.02,
        },
        knobs=knobs,
    )
    assert strong["eligible"] is True
    assert pick_tf_winner([weak, strong])["tf"] == "H5"
    assert score_tf_row(strong, knobs=knobs) > score_tf_row(weak, knobs=knobs)


def test_knobs_read_horizon_sweep():
    knobs = load_tf_sweep_knobs()
    assert knobs["min_edge_vs_breakeven"] == pytest.approx(0.03)
    assert "leaderboard.json" in knobs["leaderboard_path"]
    assert knobs["artifact_root"] == "data/dl/sweep"
    assert knobs["n_bars"] == [1, 2, 3, 4]
    assert bool(knobs["enabled"]) is False
    assert knobs["symbols"] == ["STP_500"]


def test_tf_score_settle_n_and_history_fallbacks():
    from src.application.services.deep_learning.tf_sweep_score import (
        _history_bars_for_settle,
        checkpoint_settle_eligible,
        resolve_settle_n,
    )

    assert resolve_settle_n({"deploy_settlement_n": 22}) == 22
    assert resolve_settle_n({"settle_n": "bad"}) == 0
    assert _history_bars_for_settle({}, None) == 0
    assert _history_bars_for_settle({}, "x") == 0
    assert (
        _history_bars_for_settle(
            {},
            {"deep_learning": {"training_history_bars": 900}},
        )
        == 900
    )
    assert (
        _history_bars_for_settle(
            {},
            {"deep_learning": {}, "data_handler": {"micro_history_bars": 850}},
        )
        == 850
    )
    assert (
        _history_bars_for_settle(
            {},
            {"deep_learning": {}, "data_handler": {"history_bars": 810}},
        )
        == 810
    )
    assert _history_bars_for_settle({}, {"deep_learning": {}, "data_handler": {}}) == 0
    assert (
        is_tf_eligible(
            rank_wr=0.9,
            be_implied=0.5,
            min_edge_vs_breakeven=0.03,
            settle_n=4,
            min_settle_n=16,
            history_bars=50,
            min_history_bars=100,
        )
        is False
    )
    assert (
        is_tf_eligible(
            rank_wr=0.9,
            be_implied=0.5,
            min_edge_vs_breakeven=0.03,
            settle_n=24,
            min_settle_n=16,
            history_bars=50,
            min_history_bars=100,
        )
        is False
    )
    assert (
        checkpoint_settle_eligible(
            {
                "deploy_settlement_win_rate": 0.70,
                "deploy_settlement_n": 24,
            },
            {
                "deep_learning": {
                    "horizon_sweep": {
                        "min_edge_vs_breakeven": 0.03,
                        "min_settle_n": 16,
                        "min_history_bars": 800,
                    },
                    "training_history_bars": 900,
                },
                "risk_management": {"params": {"payout_estimate": 0.72}},
            },
        )
        is True
    )
