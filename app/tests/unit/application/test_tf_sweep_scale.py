"""Testes de escala wall-clock e score/history do sweep multi-TF."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.application.services.deep_learning.tf_sweep_config import (
    load_tf_sweep_knobs,
    resolve_enabled_candidates,
)
from src.application.services.deep_learning.tf_sweep_score import is_tf_eligible


def test_tf_wallclock_scale_m2_identity():
    from src.application.services.deep_learning.tf_sweep_scale import (
        scale_bars,
        scale_lookback,
    )

    assert scale_lookback(120) == 720
    assert scale_bars(14, 120) == 14
    assert scale_bars(14, 60) == 28
    assert scale_bars(14, 300) == 6


def test_run_tf_sweep_with_mock_train(tmp_path: Path):
    from scripts.operations.sweep_train_timeframes import run_tf_sweep

    settings = {
        "data_handler": {"micro_granularity": 120, "mini_granularity": 120, "granularity": 3600},
        "deep_learning": {
            "lookback": 720,
            "tf_sweep": {
                "min_edge_vs_breakeven": 0.03,
                "artifact_root": "art",
                "leaderboard_path": "art/leaderboard.json",
            },
        },
        "risk_management": {"params": {"payout_estimate": 0.72, "duration": 2, "duration_unit": "m"}},
        "orchestrator": {
            "cycle_interval_seconds": 120,
            "signature_boundary_seconds": 120,
            "exec_empty_retry_seconds": 120,
        },
    }
    knobs = load_tf_sweep_knobs(settings)
    cands = [c for c in resolve_enabled_candidates() if c["tf"] in ("M2", "M5")]

    def _fake(patched, candidate, art: Path):
        _ = patched
        art.mkdir(parents=True, exist_ok=True)
        settle = 0.65 if candidate["tf"] == "M5" else 0.55
        return {
            "deploy_ok": True,
            "val_accuracy": settle,
            "settle_wr": settle,
            "settle_n": 24,
            "val_brier": 0.22,
            "oos_sharpness": 0.02,
            "deploy_win_rate": 0.6,
            "train_exit_code": 0,
            "error": None,
        }

    board = run_tf_sweep(
        settings=settings,
        knobs=knobs,
        candidates=cands,
        train_fn=_fake,
        repo_root=tmp_path,
    )
    assert len(board) == 2
    assert (tmp_path / "art" / "leaderboard.json").is_file()
    m5 = next(r for r in board if r["tf"] == "M5")
    assert m5["eligible"] is True
    m2 = next(r for r in board if r["tf"] == "M2")
    assert m2["eligible"] is False


def test_tf_wallclock_scale_maps_bool_float_and_nested():
    from src.application.services.deep_learning.tf_sweep_scale import apply_tf_wallclock_scale

    settings = {
        "deep_learning": {
            "indicators": {
                "windows": {"rsi_period": 14, "enabled_flag": True, "alpha": 2.0, "label": "keep"},
                "congestion": {"min_bars": 100},
                "trend_consensus": {"min_bars": 40},
            },
            "deploy_gate": {"mini_bars": 120, "max_eval_steps": 10, "min_trades": 2},
        },
        "data_handler": {},
        "orchestrator": {
            "execution": {
                "scale_vision": {"slope_bars": 5},
                "dynamic_threshold": {"baseline_lookback": 72},
            }
        },
    }
    out = apply_tf_wallclock_scale(settings, 60)
    win = out["deep_learning"]["indicators"]["windows"]
    assert win["rsi_period"] == 28
    assert win["enabled_flag"] is True
    assert win["alpha"] == pytest.approx(4.0)
    assert win["label"] == "keep"
    assert out["deep_learning"]["indicators"]["congestion"]["min_bars"] == 200
    assert out["deep_learning"]["indicators"]["trend_consensus"]["min_bars"] == 80
    assert out["deep_learning"]["deploy_gate"]["max_eval_steps"] >= 48
    with pytest.raises(ValueError, match="deep_learning"):
        apply_tf_wallclock_scale({"deep_learning": []}, 120)


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
                    "tf_sweep": {"min_edge_vs_breakeven": 0.03, "min_settle_n": 16, "min_history_bars": 800},
                    "training_history_bars": 900,
                },
                "risk_management": {"params": {"payout_estimate": 0.72}},
            },
        )
        is True
    )
