"""Testes do invert_exec_side seletivo (so ev_call)."""

from __future__ import annotations

from types import SimpleNamespace

from src.application.services.execution_invert_side import apply_invert_exec_side, invert_exec_side_enabled
from src.application.services.market_audit_cycle import format_gates_audit_line
from src.domain.models.trade import TradeDirection


def _orch(*, enabled: bool = True):
    return SimpleNamespace(config={"orchestrator": {"execution": {"invert_exec_side": enabled}}})


def test_invert_exec_side_enabled_from_ssot():
    assert invert_exec_side_enabled({}) is False
    assert invert_exec_side_enabled({"invert_exec_side": True}) is True
    assert invert_exec_side_enabled({"invert_exec_side": False}) is False


def test_apply_invert_only_on_fusion_ev_call():
    metrics = {
        "exec_direction": "CALL",
        "resolved_direction": "CALL",
        "fusion_applied": True,
        "fusion_reason": "ev_call",
        "execution_candidate_ready": True,
    }
    out = apply_invert_exec_side(metrics, TradeDirection.CALL, orch=_orch())
    assert out == TradeDirection.PUT
    assert metrics["invert_exec_side"] is True
    assert metrics["invert_from"] == "CALL"


def test_apply_invert_skips_fusion_ev_put():
    metrics = {
        "exec_direction": "PUT",
        "fusion_applied": True,
        "fusion_reason": "ev_put",
        "execution_candidate_ready": True,
    }
    out = apply_invert_exec_side(metrics, TradeDirection.PUT, orch=_orch())
    assert out == TradeDirection.PUT
    assert metrics.get("invert_exec_side") is False
    assert metrics.get("invert_skipped_reason") == "ev_put"


def test_apply_invert_clears_neg_edge_empty_on_ev_call():
    metrics = {
        "exec_direction": "CALL",
        "fusion_applied": True,
        "fusion_reason": "ev_call",
        "gate_reason": "neg_edge",
        "signal_status": "SKIP:NEG_EDGE",
        "execution_candidate_ready": False,
        "neg_edge_bootstrap_deep": True,
    }
    out = apply_invert_exec_side(metrics, TradeDirection.CALL, orch=_orch())
    assert out == TradeDirection.PUT
    assert metrics["execution_candidate_ready"] is True
    assert metrics.get("gate_reason") is None
    assert metrics.get("invert_cleared_signal_empty") is True


def test_gates_line_shows_invert_or_skip():
    line = format_gates_audit_line(
        {
            "fusion_applied": True,
            "fusion_side_pre_invert": "CALL",
            "fusion_ev_call": -0.01,
            "fusion_ev_put": -0.05,
            "fusion_p_eff": 0.61,
            "fusion_reason": "ev_call",
            "invert_exec_side": True,
            "invert_from": "CALL",
            "exec_direction": "PUT",
            "calibrated_prob": 0.55,
        }
    )
    assert "INVERT CALL->PUT" in line
    skip_line = format_gates_audit_line(
        {
            "fusion_applied": True,
            "fusion_ev_call": -0.05,
            "fusion_ev_put": 0.3,
            "fusion_p_eff": 0.79,
            "fusion_reason": "ev_put",
            "invert_exec_side": False,
            "invert_skipped_reason": "ev_put",
            "exec_direction": "PUT",
            "calibrated_prob": 0.3,
        }
    )
    assert "INVERT skip=ev_put" in skip_line


def test_invert_from_orch_branches_and_skip_reasons():
    from unittest.mock import patch

    assert apply_invert_exec_side({"fusion_applied": True, "fusion_reason": "ev_call"}, TradeDirection.CALL) == (
        TradeDirection.CALL
    )
    assert (
        apply_invert_exec_side(
            {"fusion_applied": True, "fusion_reason": "ev_call"},
            TradeDirection.CALL,
            orch=SimpleNamespace(config="bad"),
        )
        == TradeDirection.CALL
    )
    assert (
        apply_invert_exec_side(
            {"fusion_applied": True, "fusion_reason": "ev_call"},
            TradeDirection.CALL,
            orch=SimpleNamespace(config={"orchestrator": "bad"}),
        )
        == TradeDirection.CALL
    )
    metrics_nf = {"fusion_applied": False, "exec_direction": "CALL"}
    assert apply_invert_exec_side(metrics_nf, TradeDirection.CALL, orch=_orch()) == TradeDirection.CALL
    assert metrics_nf.get("invert_skipped_reason") == "no_fusion"
    metrics_empty = {"fusion_applied": True, "fusion_reason": "", "exec_direction": "CALL"}
    assert apply_invert_exec_side(metrics_empty, TradeDirection.CALL, orch=_orch()) == TradeDirection.CALL
    assert metrics_empty.get("invert_skipped_reason") == "not_ev_call"
    metrics_side = {"fusion_applied": True, "fusion_reason": "ev_call", "exec_direction": "PUT"}
    assert apply_invert_exec_side(metrics_side, TradeDirection.PUT, orch=_orch()) == TradeDirection.PUT
    assert metrics_side.get("invert_skipped_reason") == "side_not_call"
    with patch(
        "src.application.services.execution_invert_side.merge_settings_block",
        return_value={},
    ):
        assert invert_exec_side_enabled({}) is False
