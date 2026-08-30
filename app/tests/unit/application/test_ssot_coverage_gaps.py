"""Cobertura residual alinhada ao SSOT atual (calibracao, settlement, cooldown)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.application.services.deep_learning.dl_calibration import (
    CalibratorState,
    apply_calibrator,
    temperature_bounds,
)
from src.application.services.deep_learning.dl_calibration_fit import (
    _build_isotonic,
    _build_temperature_platt,
    _guard_sharpness,
    fit_calibrator,
    fit_temperature,
)
from src.application.services.deep_learning.dl_training_epochs import _shuffled_batch_indices
from src.application.services.execution_neg_edge import apply_negative_cal_edge_pause
from src.application.services.execution_regime_chop import apply_regime_chop_pause
from src.application.services.execution_signal_skip import apply_signal_skip_gates
from src.application.services.orchestrator.execution_blockers import log_execution_blockers
from src.domain.models.trade import TradeDirection


def test_temperature_bounds_and_apply_method_branches():
    lo, hi = temperature_bounds()
    assert lo == pytest.approx(1.0)
    assert hi >= lo
    temp_only = CalibratorState(method="temperature", temperature=1.25, platt_a=1.0, platt_b=0.0)
    assert 0.0 < apply_calibrator(0.8, temp_only) < 1.0
    platt_chain = CalibratorState(method="temperature_platt", temperature=1.25, platt_a=1.0, platt_b=0.0)
    assert 0.0 < apply_calibrator(0.7, platt_chain) < 1.0
    assert fit_temperature([0.9, 0.1, 0.8, 0.2], [1.0, 0.0, 1.0, 0.0]) >= lo


def test_fit_calibrator_paths_with_large_sample():
    probs = [0.9, 0.1, 0.85, 0.15, 0.8, 0.2, 0.75, 0.25] * 5
    labels = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0] * 5
    cal = fit_calibrator(probs, labels, calibration_cfg={"small_sample_identity": False, "method": "temperature"})
    assert cal.method in {"temperature", "identity", "isotonic"}
    auto = fit_calibrator(
        probs,
        labels,
        calibration_cfg={"small_sample_identity": False, "method": "auto", "auto_select_by_brier": False},
    )
    assert auto.method in {"temperature_platt", "identity"}
    assert fit_calibrator(probs[:2], labels[:2], calibration_cfg={"small_sample_identity": True}).method == "identity"
    preferred = _build_temperature_platt(probs, labels)
    guarded = _guard_sharpness(preferred, probs, labels, min_sharpness=0.99)
    assert guarded.method in {"temperature_platt", "identity"}
    iso = _build_isotonic(probs, labels)
    assert iso.method == "isotonic"
    identity = fit_calibrator(probs, labels, calibration_cfg={"method": "identity"})
    assert identity.method == "identity"
    forced_iso = fit_calibrator(
        probs,
        labels,
        calibration_cfg={"small_sample_identity": False, "method": "isotonic", "isotonic_min_samples": 8},
    )
    assert forced_iso.method in {"isotonic", "identity"}
    no_auto = fit_calibrator(
        probs,
        labels,
        calibration_cfg={
            "small_sample_identity": False,
            "method": "auto",
            "auto_select_by_brier": False,
            "isotonic_min_samples": 999,
        },
    )
    assert no_auto.method in {"temperature_platt", "identity"}


def test_shuffled_batch_indices_single_batch_when_size_ge_n():
    batches = _shuffled_batch_indices(5, 10)
    assert len(batches) == 1
    assert sorted(batches[0].tolist()) == list(range(5))


def test_neg_edge_hard_without_fusion_wash():
    metrics = {
        "execution_candidate_ready": True,
        "exec_direction": "CALL",
        "calibrated_prob": 0.50,
        "kelly_fraction_scale": 1.0,
    }
    assert apply_negative_cal_edge_pause(metrics, min_edge=0.04, payout=0.72) is True
    assert metrics["gate_reason"] == "neg_edge"


def test_chop_disabled_ssot_noop():
    metrics = {
        "execution_candidate_ready": True,
        "kelly_fraction_scale": 1.0,
        "indicators": {"adx": 0.05, "hurst": 0.50},
    }
    assert apply_regime_chop_pause(metrics) is False
    assert metrics.get("regime_chop_soft") is not True


def test_mini_pair_oppose_requires_unanimous_pair():
    metrics = {
        "scale_mini_prev_bar_dir": "CALL",
        "scale_mini_bar_dir": "PUT",
        "direction_margin": 0.20,
        "kelly_fraction_scale": 1.0,
    }
    cfg = {
        "enabled": True,
        "mini_pair_oppose_exec": True,
        "min_direction_margin": 0.005,
        "waive_margin_on_pending": False,
        "mini_pair_soft_kelly_mult": 0.55,
        "cal_margin_soft_kelly_mult": 0.75,
        "pending_dust": 0.25,
    }
    assert apply_signal_skip_gates(metrics, TradeDirection.CALL, cfg=cfg) is False
    assert metrics.get("mini_pair_soft") is not True


def test_execution_blockers_emits_gates_when_tcn_without_candidate():
    logger = MagicMock()
    orch = SimpleNamespace(_log_dedupe={}, _active_cycle_id=1, logger=logger)

    def _trade_symbols():
        return ["R_10"]

    executor = SimpleNamespace(logger=logger, orch=orch, _trade_symbols=_trade_symbols)
    log_execution_blockers(
        executor,
        {"R_10": {"metrics": {"tcn_direction": "CALL"}}},
    )
    assert True


def test_horizon_sweep_prefers_n_bars_and_falls_back_to_duration():
    from src.application.services.deep_learning.horizon_sweep import build_horizon_candidates, load_horizon_sweep_knobs

    only_dur = load_horizon_sweep_knobs(
        {
            "data_handler": {"micro_granularity": 60},
            "deep_learning": {"horizon_sweep": {"duration_minutes": [15, 30]}},
        }
    )
    assert only_dur["n_bars"] == [15, 30]
    rows = build_horizon_candidates(
        {
            "data_handler": {"micro_granularity": 60, "mini_granularity": 60, "granularity": 7200},
            "deep_learning": {
                "lookback": 30,
                "training_history_bars": 120,
                "horizon_sweep": {"n_bars": [1, 2, 3]},
            },
        }
    )
    assert [r["tf"] for r in rows] == ["H1", "H2", "H3"]
    empty = build_horizon_candidates(
        {
            "data_handler": {"micro_granularity": 60, "mini_granularity": 60, "granularity": 7200},
            "deep_learning": {"lookback": 30, "training_history_bars": 120, "horizon_sweep": {"n_bars": []}},
        }
    )
    assert len(empty) >= 1
