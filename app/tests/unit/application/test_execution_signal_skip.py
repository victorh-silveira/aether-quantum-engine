"""Testes do catalogo minimo de atenuacao de sinal (escopo 1.1)."""

from unittest.mock import MagicMock

import pytest

from src.application.services.execution_signal_skip import (
    apply_signal_skip_gates,
    is_skip_signal_status,
    metrics_block_execution,
    parse_signal_skip_config,
)
from src.application.services.orchestrator.execution_blockers import _candidate_block_reason
from src.domain.models.trade import TradeDirection


def test_parse_signal_skip_from_ssot():
    cfg = parse_signal_skip_config({})
    assert cfg["enabled"] is True
    assert cfg["min_direction_margin"] == 0.022
    assert cfg["waive_margin_on_pending"] is True
    assert cfg["mini_pair_oppose_exec"] is True
    assert cfg["waive_mini_pair_min_margin"] == 0.0
    assert cfg["mini_pair_soft_kelly_mult"] == 0.55
    assert cfg["cal_margin_soft_kelly_mult"] == 0.55
    assert cfg["pending_dust"] == 0.25
    assert cfg["chop_pause_enabled"] is True
    assert cfg["chop_adx_max"] == 0.22
    assert cfg["chop_hurst_min"] == 0.47
    assert cfg["chop_hurst_max"] == 0.53
    assert cfg["chop_soft_kelly_mult"] == 0.55
    assert cfg["neg_edge_soft_kelly_mult"] == 0.55
    assert cfg["neg_edge_hard_skip"] is True
    assert cfg["neg_edge_soft_when_closed_candle_agree"] is True
    assert cfg["neg_edge_soft_min_edge"] == pytest.approx(-0.05)
    assert "direction_loss_lock_min" not in cfg
    assert "direction_loss_toxic_escape" not in cfg
    assert "calib_gray_margin_floor" not in cfg
    assert "calib_gray_soft_kelly_mult" not in cfg
    assert "calib_gray_max_stake_pct" not in cfg
    with pytest.raises(ValueError, match="mini_pair_soft_kelly_mult"):
        parse_signal_skip_config({"mini_pair_soft_kelly_mult": 0.0})
    with pytest.raises(ValueError, match="cal_margin_soft_kelly_mult"):
        parse_signal_skip_config({"cal_margin_soft_kelly_mult": 0.0})
    with pytest.raises(ValueError, match="chop_soft_kelly_mult"):
        parse_signal_skip_config({"chop_soft_kelly_mult": 0.0})
    with pytest.raises(ValueError, match="neg_edge_soft_kelly_mult"):
        parse_signal_skip_config({"neg_edge_soft_kelly_mult": 0.0})
    with pytest.raises(ValueError, match="chop_hurst_max"):
        parse_signal_skip_config({"chop_hurst_min": 0.60, "chop_hurst_max": 0.40})


def test_metrics_block_execution_covers_prefixed_skip_and_ready():
    assert is_skip_signal_status("SKIP:LOSS_CLF_VETO") is True
    assert is_skip_signal_status("SKIP:CAL_MARGIN") is True
    assert is_skip_signal_status("OK") is False
    assert metrics_block_execution({"signal_status": "SKIP:LOSS_CLF_VETO"}) is True
    assert metrics_block_execution({"execution_candidate_ready": False}) is True
    assert metrics_block_execution({"gate_reason": "cal_margin"}) is False
    assert metrics_block_execution({"signal_skip_reason": "mini_pair_oppose"}) is False
    assert metrics_block_execution({"gate_reason": "loss_clf_veto"}) is False
    assert metrics_block_execution({"execution_candidate_ready": True, "signal_status": "OPEN"}) is False


def test_mini_pair_oppose_always_soft_kelly():
    metrics = {
        "scale_mini_prev_bar_dir": "PUT",
        "scale_mini_bar_dir": "PUT",
        "direction_margin": 0.029,
        "kelly_fraction_scale": 1.0,
    }
    assert apply_signal_skip_gates(metrics, TradeDirection.CALL) is False
    assert metrics.get("gate_reason") is None
    assert metrics.get("execution_candidate_ready") is not False
    assert metrics["signal_skip_waived"] == "mini_pair_soft"
    assert metrics["mini_pair_soft"] is True
    assert metrics["kelly_fraction_scale"] == pytest.approx(0.55)
    assert metrics_block_execution(metrics) is False


def test_mini_pair_oppose_strong_margin_soft_waive():
    metrics = {
        "scale_mini_prev_bar_dir": "PUT",
        "scale_mini_bar_dir": "PUT",
        "direction_margin": 0.08,
        "kelly_fraction_scale": 1.0,
    }
    assert apply_signal_skip_gates(metrics, TradeDirection.CALL) is False
    assert metrics.get("gate_reason") is None
    assert metrics.get("execution_candidate_ready") is not False
    assert metrics["signal_skip_waived"] == "mini_pair_soft"
    assert metrics["mini_pair_soft"] is True
    assert metrics["kelly_fraction_scale"] == 0.55


def test_cal_margin_soft_when_margin_weak_explore():
    metrics = {
        "scale_mini_prev_bar_dir": "CALL",
        "scale_mini_bar_dir": "CALL",
        "direction_margin": 0.020,
        "pending_loss_total": 0.0,
        "kelly_fraction_scale": 1.0,
    }
    assert apply_signal_skip_gates(metrics, TradeDirection.CALL) is False
    assert metrics.get("gate_reason") is None
    assert metrics["signal_skip_waived"] == "cal_margin_soft"
    assert metrics["cal_margin_soft"] is True
    assert metrics["kelly_fraction_scale"] == pytest.approx(0.55)
    assert metrics_block_execution(metrics) is False


def test_cal_margin_waived_when_pending_material():
    metrics = {
        "scale_mini_prev_bar_dir": "CALL",
        "scale_mini_bar_dir": "CALL",
        "direction_margin": 0.020,
        "pending_loss_total": 27.0,
    }
    assert apply_signal_skip_gates(metrics, TradeDirection.CALL) is False
    assert metrics.get("signal_skip_waived") == "cal_margin_pending"
    assert metrics.get("gate_reason") is None


def test_pending_map_fallback_and_bad_margin():
    metrics = {
        "scale_mini_prev_bar_dir": "CALL",
        "scale_mini_bar_dir": "PUT",
        "direction_margin": object(),
        "pending_loss_total": object(),
    }
    orch = MagicMock()
    del orch.risk_manager.pending_loss_total
    type(orch.risk_manager).pending_loss_total = property(lambda self: None)
    orch.risk_manager.pending_loss = {"R_10": 12.0}
    assert apply_signal_skip_gates(metrics, TradeDirection.CALL, orch=orch) is False
    assert metrics.get("signal_skip_waived") == "cal_margin_pending"


def test_pending_total_edge_paths():
    from src.application.services.execution_signal_skip import _pending_total

    assert _pending_total({"pending_loss_total": "x"}, None) == 0.0
    orch_none_rm = MagicMock()
    orch_none_rm.risk_manager = None
    assert _pending_total({}, orch_none_rm) == 0.0

    orch_bad_fn = MagicMock()

    def _boom():
        raise ValueError("bad")

    orch_bad_fn.risk_manager.pending_loss_total = _boom
    assert _pending_total({}, orch_bad_fn) == 0.0

    orch_bad_map = MagicMock()
    orch_bad_map.risk_manager.pending_loss_total = "not_callable"
    orch_bad_map.risk_manager.pending_loss = {"R_10": object()}
    assert _pending_total({}, orch_bad_map) == 0.0

    orch_empty = MagicMock()
    orch_empty.risk_manager.pending_loss_total = "x"
    orch_empty.risk_manager.pending_loss = None
    assert _pending_total({}, orch_empty) == 0.0


def test_mini_pair_oppose_disabled_falls_to_cal_soft():
    metrics = {
        "scale_mini_prev_bar_dir": "PUT",
        "scale_mini_bar_dir": "PUT",
        "direction_margin": 0.010,
        "pending_loss_total": 0.0,
        "kelly_fraction_scale": 1.0,
    }
    assert (
        apply_signal_skip_gates(
            metrics,
            TradeDirection.CALL,
            cfg={
                "enabled": True,
                "min_direction_margin": 0.022,
                "waive_margin_on_pending": True,
                "mini_pair_oppose_exec": False,
                "cal_margin_soft_kelly_mult": 0.55,
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
    metrics = {
        "scale_mini_prev_bar_dir": "PUT",
        "scale_mini_bar_dir": "PUT",
        "direction_margin": 0.01,
    }
    assert apply_signal_skip_gates(metrics, TradeDirection.CALL, force=True) is False


def test_disabled_catalog_noop():
    metrics = {
        "scale_mini_prev_bar_dir": "PUT",
        "scale_mini_bar_dir": "PUT",
        "direction_margin": 0.01,
    }
    assert (
        apply_signal_skip_gates(
            metrics,
            TradeDirection.CALL,
            cfg={
                "enabled": False,
                "min_direction_margin": 0.022,
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
        "direction_margin": 0.018,
    }
    orch = MagicMock()
    orch.risk_manager.pending_loss_total.return_value = 40.0
    assert apply_signal_skip_gates(metrics, TradeDirection.CALL, orch=orch) is False
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
        "direction_margin": 0.009,
        "edge": -0.008,
        "kelly_fraction_scale": 1.0,
        "exec_direction": "CALL",
    }
    assert apply_signal_skip_gates(metrics, TradeDirection.CALL) is False
    assert metrics["cal_margin_soft"] is True
    assert metrics["signal_skip_waived"] == "cal_margin_soft"
    assert metrics["kelly_fraction_scale"] == pytest.approx(0.55)
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
    assert apply_signal_skip_gates(metrics, TradeDirection.CALL, orch=orch, symbol="R_10") is False
    assert metrics["exec_direction"] == "CALL"
    assert metrics["resolved_direction"] == "CALL"
    assert "dir_lock_flip_from" not in metrics
    assert metrics.get("signal_skip_waived") == "mini_pair_soft"
    assert metrics["kelly_fraction_scale"] == pytest.approx(0.55)
    assert metrics.get("gate_reason") is None
    assert metrics_block_execution(metrics) is False
    reset_direction_persistence_tracker()
