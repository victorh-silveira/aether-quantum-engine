"""Testes do catalogo minimo de atenuacao de sinal (parte 3)."""

from unittest.mock import MagicMock

import pytest

from src.application.services.execution_signal_skip import (
    apply_signal_skip_gates,
    metrics_block_execution,
    parse_signal_skip_config,
)
from src.application.services.orchestrator.execution_blockers import _candidate_block_reason
from src.domain.models.trade import TradeDirection


def _mini_pair_cfg(**overrides):
    cfg = parse_signal_skip_config(
        {
            "mini_pair_oppose_exec": True,
            "mini_pair_soft_kelly_mult": 0.55,
            "cal_margin_soft_kelly_mult": 0.55,
            "min_direction_margin": 0.022,
            "waive_margin_on_pending": False,
        }
    )
    cfg.update(overrides)
    return cfg


def test_mini_pair_oppose_disabled_falls_to_cal_soft():
    metrics = {
        "scale_mini_prev_bar_dir": "PUT",
        "scale_mini_bar_dir": "PUT",
        "direction_margin": 0.002,
        "pending_loss_total": 0.0,
        "kelly_fraction_scale": 1.0,
    }
    assert (
        apply_signal_skip_gates(
            metrics,
            TradeDirection.CALL,
            cfg={
                "enabled": True,
                "min_direction_margin": 0.005,
                "waive_margin_on_pending": True,
                "mini_pair_oppose_exec": False,
                "cal_margin_soft_kelly_mult": 0.75,
                "pending_dust": 0.25,
            },
        )
        is False
    )
    assert metrics["signal_skip_waived"] == "cal_margin_soft"
    assert metrics.get("gate_reason") is None


def test_adapted_explosion_high_margin_passes():
    metrics = {
        "scale_mini_prev_bar_dir": "PUT",
        "scale_mini_bar_dir": "PUT",
        "direction_margin": 0.030,
        "scale_adapted": True,
        "kelly_fraction_scale": 1.0,
    }
    assert apply_signal_skip_gates(metrics, TradeDirection.PUT) is False
    assert metrics.get("gate_reason") is None


def test_force_bypasses_signal_skip():
    metrics = {"scale_mini_prev_bar_dir": "PUT", "scale_mini_bar_dir": "PUT", "direction_margin": 0.01}
    assert apply_signal_skip_gates(metrics, TradeDirection.CALL, force=True) is False


def test_disabled_catalog_noop():
    metrics = {"scale_mini_prev_bar_dir": "PUT", "scale_mini_bar_dir": "PUT", "direction_margin": 0.01}
    assert (
        apply_signal_skip_gates(
            metrics,
            TradeDirection.CALL,
            cfg={
                "enabled": False,
                "min_direction_margin": 0.005,
                "waive_margin_on_pending": True,
                "mini_pair_oppose_exec": True,
                "pending_dust": 0.25,
            },
        )
        is False
    )


def test_pending_from_orch_risk_manager():
    metrics = {
        "scale_mini_prev_bar_dir": "CALL",
        "scale_mini_bar_dir": "CALL",
        "direction_margin": 0.002,
    }
    orch = MagicMock()
    orch.risk_manager.pending_loss_total.return_value = 40.0
    cfg = parse_signal_skip_config({"waive_margin_on_pending": True})
    assert apply_signal_skip_gates(metrics, TradeDirection.CALL, orch=orch, cfg=cfg) is False
    assert metrics.get("signal_skip_waived") == "cal_margin_pending"


def test_candidate_block_reason_ignores_signal_soft_reasons():
    assert _candidate_block_reason({"gate_reason": "mini_pair_oppose"}) is None
    assert _candidate_block_reason({"gate_reason": "cal_margin"}) is None
    assert _candidate_block_reason({"signal_skip_reason": "cal_margin", "gate_reason": None}) is None
    assert _candidate_block_reason({"gate_reason": "training"}) == "training"
    assert _candidate_block_reason({"gate_reason": "neg_edge"}) == "neg_edge"
    assert _candidate_block_reason({"gate_reason": "loss_clf_veto"}) is None
    assert _candidate_block_reason({"execution_candidate_ready": True}) == "ready_not_selected"


def test_weak_cal_margin_uses_cal_margin_soft():
    metrics = {
        "direction_margin": 0.002,
        "edge": -0.008,
        "kelly_fraction_scale": 1.0,
        "exec_direction": "CALL",
    }
    assert apply_signal_skip_gates(metrics, TradeDirection.CALL) is False
    assert metrics["cal_margin_soft"] is True
    assert metrics["signal_skip_waived"] == "cal_margin_soft"
    assert metrics["kelly_fraction_scale"] == pytest.approx(0.75)
    assert "calib_gray_soft" not in metrics


def test_direction_loss_lock_removed_keeps_tcn_side():
    from src.application.services.direction_loss_tracker import (
        record_direction_outcome,
        reset_direction_persistence_tracker,
    )

    reset_direction_persistence_tracker()
    record_direction_outcome("R_10", "CALL", won=False)
    record_direction_outcome("R_10", "CALL", won=False)
    metrics = {
        "scale_mini_prev_bar_dir": "PUT",
        "scale_mini_bar_dir": "PUT",
        "direction_margin": 0.12,
        "kelly_fraction_scale": 1.0,
        "exec_direction": "CALL",
        "resolved_direction": "CALL",
    }
    orch = MagicMock()
    orch._log_dedupe = {}
    assert (
        apply_signal_skip_gates(metrics, TradeDirection.CALL, orch=orch, symbol="R_10", cfg=_mini_pair_cfg()) is False
    )
    assert metrics["exec_direction"] == "CALL"
    assert metrics["resolved_direction"] == "CALL"
    assert "dir_lock_flip_from" not in metrics
    assert metrics.get("signal_skip_waived") == "mini_pair_soft"
    assert metrics["kelly_fraction_scale"] == pytest.approx(0.55)
    assert metrics.get("gate_reason") is None
    assert metrics_block_execution(metrics) is False


def test_infer_dl_direction_abstained_on_neutral_zone():
    from src.application.services.execution_direction_checks import infer_dl_direction

    assert infer_dl_direction({"metrics": {"calibration_mode": "neutral_zone"}}) is None
    assert infer_dl_direction({"metrics": {"gate_reason": "neutral_zone"}}) is None
    assert infer_dl_direction({"direction": TradeDirection.CALL, "metrics": {}}) == TradeDirection.CALL
