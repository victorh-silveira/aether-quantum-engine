"""Testes de nitidez TCN, anti-colapso de calibracao e coerencia Edge/margem."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.application.services.deep_learning.dl_calibration import CalibratorState, apply_calibrator_stable
from src.application.services.deep_learning.dl_calibration_fit import fit_calibrator
from src.application.services.deep_learning.dl_cycle_log import log_dl_cycle_summary
from src.application.services.deep_learning.dl_sharpness import (
    assert_export_sharpness_floor,
    assert_export_sharpness_value,
    mean_sharpness,
    sharpness_pass_fraction,
)
from src.application.services.execution_quality_gate import passes_execution_quality
from src.application.services.execution_quality_gate_margin import stamp_edge_without_direction
from src.application.services.market_audit_log_helpers import cluster_symbol_token
from src.domain.models.trade import TradeDirection


def test_mean_sharpness_and_pass_fraction():
    probs = [0.40, 0.50, 0.60]
    assert mean_sharpness(probs) == pytest.approx(0.066666, abs=1e-5)
    assert mean_sharpness([]) == 0.0
    assert sharpness_pass_fraction([], floor=0.03) == 0.0
    assert sharpness_pass_fraction(probs, floor=0.03) == pytest.approx(2.0 / 3.0)


def test_assert_export_sharpness_floor_accepts_sharp_probs():
    assert assert_export_sharpness_floor([0.40, 0.60, 0.35, 0.70], floor=0.03) >= 0.03


def test_assert_export_sharpness_blocks_collapsed():
    with pytest.raises(RuntimeError, match="Export TCN bloqueado"):
        assert_export_sharpness_floor([0.49, 0.51, 0.50], floor=0.03)
    with pytest.raises(RuntimeError, match="Export TCN bloqueado"):
        assert_export_sharpness_value(0.01, floor=0.03, label="holdout")
    assert assert_export_sharpness_value(0.05, floor=0.03) == pytest.approx(0.05)


def test_select_best_calibrator_falls_back_to_sharpest():
    from src.application.services.deep_learning.dl_calibration_fit import _select_best_calibrator

    weak = CalibratorState(method="platt", temperature=1.0, platt_a=0.1, platt_b=0.0)
    stronger = CalibratorState(method="temperature_platt", temperature=1.0, platt_a=1.0, platt_b=0.0)
    chosen = _select_best_calibrator(
        [(weak, 0.10, 0.05, 0.01), (stronger, 0.12, 0.06, 0.02)],
        min_sharpness=0.03,
    )
    assert chosen is stronger


def test_apply_calibrator_stable_keeps_raw_when_calibration_collapses_margin():
    cal = CalibratorState(method="temperature_platt", temperature=2.5, platt_a=0.2, platt_b=0.0)
    raw = 0.56
    kept = apply_calibrator_stable(raw, cal, margin_floor=0.03)
    assert abs(kept - 0.5) + 1e-12 >= 0.03
    assert kept == pytest.approx(raw)


def test_fit_calibrator_prefers_eligible_sharpness():
    probs = [0.1, 0.2, 0.8, 0.9, 0.15, 0.85]
    labels = [0.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    cal = fit_calibrator(
        probs,
        labels,
        calibration_cfg={
            "method": "auto",
            "auto_select_by_brier": True,
            "isotonic_min_samples": 20,
            "min_calibration_sharpness": 0.03,
        },
    )
    calibrated = [apply_calibrator_stable(p, cal, margin_floor=0.0) for p in probs]
    assert mean_sharpness(calibrated) + 1e-12 >= 0.02


def test_stamp_edge_without_direction_squeezes_score():
    metrics = {"trade_score": 0.80, "conviction": 0.80, "predicted_payoff_edge": 0.12}
    stamp_edge_without_direction(metrics, margin_floor=0.03, score_factor=0.85)
    assert metrics["edge_without_direction"] is True
    assert metrics["trade_score"] == pytest.approx(0.68)
    assert metrics["edge_without_direction_penalty"] == pytest.approx(0.12)


def test_stamp_edge_without_direction_without_scores():
    metrics = {"predicted_payoff_edge": 0.05}
    stamp_edge_without_direction(metrics, margin_floor=0.03)
    assert metrics["edge_without_direction"] is True
    assert "trade_score" not in metrics


def test_passes_execution_quality_stamps_edge_without_direction():
    metrics = {
        "calibrated_prob": 0.489,
        "exec_direction": "PUT",
        "predicted_payoff_edge": 0.08,
        "trade_score": 0.70,
        "conviction": 0.70,
        "val_accuracy": 0.55,
        "deploy_ok": True,
    }
    ok = passes_execution_quality(
        metrics,
        exec_cfg={"quality_gate": {"min_direction_margin": 0.03, "regular": {"min_direction_margin": 0.03}}},
        risk_manager=None,
    )
    assert ok is False
    assert metrics["quality_gate_reason"] == "direction_margin_gate"
    assert metrics["edge_without_direction"] is True
    assert metrics["trade_score"] < 0.70


def test_cluster_symbol_token_invalid_margin_falls_back():
    token = cluster_symbol_token(
        "R_10",
        {
            "direction": TradeDirection.CALL,
            "metrics": {
                "raw_prob": 0.60,
                "calibrated_prob": 0.62,
                "direction_margin": object(),
                "predicted_payoff_edge": 0.04,
            },
        },
    )
    assert "Margin: 0.120" in token
    assert "Edge: +0.209" in token


def test_cluster_symbol_token_empty_symbol():
    assert cluster_symbol_token(None) == "N/A"
    assert cluster_symbol_token("") == "N/A"


def test_cluster_symbol_token_no_entry():
    assert cluster_symbol_token("R_10") == "R_10"
    assert cluster_symbol_token("R_10", None) == "R_10"


def test_log_dl_cycle_summary_emits_calib_gray():
    logger = MagicMock()
    orch = SimpleNamespace(_active_cycle_id=7, config={"data_handler": {"micro_granularity": 120}})
    decisions = {
        "R_10": {
            "direction": TradeDirection.PUT,
            "metrics": {
                "raw_prob": 0.486,
                "calibrated_prob": 0.489,
                "cal_margin": 0.011,
                "raw_margin": 0.014,
                "quality_min_direction_margin": 0.03,
                "calibrator_method": "temperature_platt",
                "calibrator_temperature": 1.8,
                "calibrator_platt_a": 0.9,
                "calibrator_platt_b": 0.0,
                "predicted_payoff_edge": 0.05,
            },
        }
    }
    log_dl_cycle_summary(logger, decisions, recovery_active=False, pending_loss_total=0.0, orch=orch)
    assert logger.info.call_count >= 1


def test_log_dl_cycle_summary_skips_calib_gray_when_margin_ok():
    logger = MagicMock()
    orch = SimpleNamespace(_active_cycle_id=8, config={"data_handler": {"micro_granularity": 120}})
    decisions = {
        "R_10": {
            "direction": TradeDirection.CALL,
            "metrics": {
                "raw_prob": 0.60,
                "calibrated_prob": 0.62,
                "cal_margin": 0.12,
                "quality_min_direction_margin": 0.03,
                "predicted_payoff_edge": 0.10,
            },
        }
    }
    log_dl_cycle_summary(logger, decisions, recovery_active=False, pending_loss_total=0.0, orch=orch)
    assert all("CALIB_GRAY" not in str(call) for call in logger.info.call_args_list)
