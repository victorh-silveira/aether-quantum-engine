"""Testes do catalogo minimo de SKIP de sinal (escopo 1.1)."""

from unittest.mock import MagicMock

from src.application.services.execution_signal_skip import (
    apply_signal_skip_gates,
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
    assert cfg["pending_dust"] == 0.25


def test_mini_pair_oppose_c3_like_skips_call():
    metrics = {
        "scale_mini_prev_bar_dir": "PUT",
        "scale_mini_bar_dir": "PUT",
        "direction_margin": 0.029,
    }
    assert apply_signal_skip_gates(metrics, TradeDirection.CALL) is True
    assert metrics["execution_candidate_ready"] is False
    assert metrics["gate_reason"] == "mini_pair_oppose"
    assert metrics["signal_skip_reason"] == "mini_pair_oppose"


def test_cal_margin_skips_explore_when_margin_weak():
    metrics = {
        "scale_mini_prev_bar_dir": "CALL",
        "scale_mini_bar_dir": "CALL",
        "direction_margin": 0.020,
        "pending_loss_total": 0.0,
    }
    assert apply_signal_skip_gates(metrics, TradeDirection.CALL) is True
    assert metrics["gate_reason"] == "cal_margin"


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


def test_mini_pair_oppose_disabled_falls_to_margin():
    metrics = {
        "scale_mini_prev_bar_dir": "PUT",
        "scale_mini_bar_dir": "PUT",
        "direction_margin": 0.010,
        "pending_loss_total": 0.0,
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
                "pending_dust": 0.25,
            },
        )
        is True
    )
    assert metrics["gate_reason"] == "cal_margin"


def test_adapted_explosion_high_margin_passes():
    metrics = {
        "scale_mini_prev_bar_dir": "PUT",
        "scale_mini_bar_dir": "PUT",
        "direction_margin": 0.030,
        "scale_adapted": True,
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


def test_candidate_block_reason_reports_signal_skip():
    assert _candidate_block_reason({"gate_reason": "mini_pair_oppose"}) == "mini_pair_oppose"
    assert _candidate_block_reason({"gate_reason": "cal_margin"}) == "cal_margin"
    assert _candidate_block_reason({"signal_skip_reason": "cal_margin", "gate_reason": None}) == "cal_margin"


def test_resolve_with_orch_applies_cal_margin_skip():
    from unittest.mock import patch

    from src.application.services.execution_direction_resolver import resolve_execution_direction
    from src.domain.models.trade import TradeDirection

    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "execute": True,
            "deploy_ok": True,
            "raw_prob": 0.51,
            "calibrated_prob": 0.51,
            "val_accuracy": 0.70,
            "predicted_payoff_edge": 0.06,
            "meta_classifier_applied": True,
            "scale_mini_prev_bar_dir": "CALL",
            "scale_mini_bar_dir": "CALL",
        },
    }
    orch = MagicMock()
    orch.config = {"infra": {"loss_classifier": {"enabled": False}}}
    orch._active_cycle_id = 0
    orch.risk_manager.pending_loss_total.return_value = 0.0
    with patch(
        "src.application.services.execution_direction_resolver.apply_loss_classifier_gate",
        return_value=False,
    ):
        result = resolve_execution_direction(entry, symbol="R_10", orch=orch)
    assert result is not None
    _dir, metrics = result
    assert metrics.get("gate_reason") == "cal_margin"
    assert metrics.get("execution_candidate_ready") is False
