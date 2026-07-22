from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.application.services.execution_direction_checks import initial_direction_checks
from src.application.services.execution_direction_resolver import (
    _apply_persistence_guard_skip,
    resolve_execution_direction,
)
from src.application.services.execution_quality_gate import passes_execution_quality
from src.application.services.execution_quality_gate_fallback import (
    cluster_quality_gate_blocks_mandatory_fallback,
)
from src.application.services.force_trade_mode import (
    force_trade_every_cycle,
    force_trade_from_config,
    force_trade_from_orch,
    resolve_force_min_stake,
    synthesize_force_direction,
    synthesize_force_trade_candidate,
)
from src.domain.models.trade import TradeDirection


def test_force_trade_flag_reads_exec_cfg():
    assert force_trade_every_cycle({"force_trade_every_cycle": True}) is True
    assert force_trade_every_cycle({"force_trade_every_cycle": False}) is False
    assert force_trade_every_cycle({}) is False
    assert force_trade_every_cycle(None) is False


def test_force_trade_from_config_and_orch_guards():
    assert force_trade_from_config(None) is False
    assert force_trade_from_config({"orchestrator": "bad"}) is False
    assert force_trade_from_config({"orchestrator": {"execution": {"force_trade_every_cycle": True}}}) is True
    assert force_trade_from_orch(None) is False
    assert force_trade_from_orch(SimpleNamespace(config=None)) is False
    assert (
        force_trade_from_orch(
            SimpleNamespace(config={"orchestrator": {"execution": {"force_trade_every_cycle": True}}})
        )
        is True
    )


def test_resolve_force_min_stake_branches():
    assert resolve_force_min_stake(None) == pytest.approx(1.0)
    assert resolve_force_min_stake({}) == pytest.approx(1.0)
    assert resolve_force_min_stake({"risk_management": "x"}) == pytest.approx(1.0)
    assert resolve_force_min_stake({"risk_management": {}}) == pytest.approx(1.0)
    assert resolve_force_min_stake({"risk_management": {"params": {"stake_min": 0.5}}}) == pytest.approx(0.5)
    assert resolve_force_min_stake({"risk_management": {"params": {"stake_min": "bad"}}}) == pytest.approx(1.0)


def test_passes_execution_quality_force_ignores_margin():
    metrics = {"direction_margin": 0.01, "calibrated_prob": 0.52, "raw_prob": 0.52}
    exec_cfg = {
        "force_trade_every_cycle": True,
        "quality_gate": {"min_direction_margin": 0.03},
    }
    assert passes_execution_quality(metrics, exec_cfg=exec_cfg) is True
    assert metrics.get("quality_guard_reject") is None


def test_initial_direction_checks_force_clears_neutral_clamp():
    entry = {
        "direction": None,
        "metrics": {
            "calibration_mode": "neutral_clamp",
            "gate_reason": "neutral_clamp",
            "raw_prob": 0.523,
            "calibrated_prob": 0.523,
        },
    }
    result = initial_direction_checks(entry, {"force_trade_every_cycle": True})
    assert result is not None
    dl_dir, _metrics, prob = result
    assert dl_dir == TradeDirection.CALL
    assert prob == 0.523


def test_initial_direction_checks_force_synthesizes_when_direction_missing():
    entry = {
        "direction": None,
        "metrics": {
            "calibration_mode": "calibrated",
            "raw_prob": 0.41,
            "calibrated_prob": 0.41,
            "execute": True,
            "deploy_ok": True,
        },
    }
    with patch(
        "src.application.services.execution_direction_checks.infer_dl_direction",
        return_value=None,
    ):
        result = initial_direction_checks(entry, {"force_trade_every_cycle": True})
    assert result is not None
    dl_dir, _metrics, _prob = result
    assert dl_dir == TradeDirection.PUT


def test_persistence_guard_skip_noop_when_force():
    entry = {"metrics": {}}
    metrics = {}
    skipped = _apply_persistence_guard_skip(
        entry,
        metrics,
        TradeDirection.CALL,
        symbol="R_10",
        peer_entry=None,
        cycle_id=1,
        infra_cfg={},
        force=True,
    )
    assert skipped == TradeDirection.CALL
    assert metrics.get("persistence_guard_skip") is None


def test_synthesize_force_candidate_call_from_raw_prob():
    decisions = {
        "R_10": {
            "direction": None,
            "metrics": {"gate_reason": "data"},
        },
        "R_50": {
            "direction": None,
            "metrics": {"raw_prob": 0.523, "calibrated_prob": 0.523},
        },
    }
    candidate = synthesize_force_trade_candidate(["R_10", "R_50"], decisions)
    assert candidate is not None
    symbol, direction, metrics = candidate
    assert symbol == "R_50"
    assert direction == TradeDirection.CALL
    assert metrics["execute"] is True
    assert metrics["trade_score"] >= 0.51


def test_synthesize_force_direction_put_below_half():
    entry = {"metrics": {"raw_prob": 0.41}}
    assert synthesize_force_direction(entry) == TradeDirection.PUT


def test_synthesize_force_direction_edge_branches():
    assert synthesize_force_direction({"metrics": {"deploy_ok": False, "raw_prob": 0.6}}) is None
    assert synthesize_force_direction({"metrics": {"gate_reason": "predict_error", "raw_prob": 0.6}}) is None
    assert synthesize_force_direction({"direction": TradeDirection.PUT, "metrics": {}}) == TradeDirection.PUT
    assert synthesize_force_direction({"metrics": {"raw_prob": "bad"}}) is None
    assert synthesize_force_direction({"metrics": {}}) is None


def test_synthesize_force_trade_candidate_guards_and_neutral_clamp():
    assert synthesize_force_trade_candidate(["R_10"], None) is None
    assert synthesize_force_trade_candidate(["R_10"], {"R_10": "bad"}) is None
    assert synthesize_force_trade_candidate(["R_10"], {"R_10": {"metrics": {"deploy_ok": False}}}) is None
    decisions = {
        "R_10": {
            "direction": None,
            "metrics": {
                "raw_prob": 0.62,
                "gate_reason": "neutral_clamp",
                "calibration_mode": "neutral_clamp",
            },
        }
    }
    candidate = synthesize_force_trade_candidate(["R_10"], decisions)
    assert candidate is not None
    _symbol, direction, metrics = candidate
    assert direction == TradeDirection.CALL
    assert metrics.get("gate_reason") is None
    assert metrics.get("calibration_mode") == "calibrated"


def test_synthesize_force_trade_does_not_emit_side_eq_flip():
    decisions = {
        "R_10": {
            "direction": TradeDirection.PUT,
            "metrics": {"raw_prob": 0.42, "calibrated_prob": 0.42},
        }
    }
    with patch("src.application.services.side_equilibrium_gate.logger") as mock_logger:
        candidate = synthesize_force_trade_candidate(["R_10"], decisions, orch=SimpleNamespace(_active_cycle_id=1))
    assert candidate is not None
    assert candidate[1] == TradeDirection.PUT
    flip_calls = [c for c in mock_logger.info.call_args_list if c.args and str(c.args[0]).startswith("SIDE_EQ_FLIP")]
    assert flip_calls == []
    assert candidate[2].get("side_eq_flipped") is not True


def test_cluster_quality_fallback_never_blocks_when_force():
    decisions = {
        "R_10": {
            "direction": TradeDirection.CALL,
            "metrics": {
                "quality_gate_reason": "adx_starvation",
                "quality_guard_reject": True,
                "raw_prob": 0.55,
            },
        }
    }
    blocked = cluster_quality_gate_blocks_mandatory_fallback(
        decisions,
        exec_cfg={"force_trade_every_cycle": True},
        risk_manager=SimpleNamespace(consecutive_losses_linear=3, pending_loss={"x": 1.0}),
        trade_symbols=["R_10"],
    )
    assert blocked is False


def test_resolve_execution_direction_force_with_weak_margin():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "raw_prob": 0.52,
            "calibrated_prob": 0.52,
            "trade_score": 0.52,
            "val_accuracy": 0.51,
            "deploy_ok": True,
        },
    }
    resolved = resolve_execution_direction(
        entry,
        exec_cfg={
            "force_trade_every_cycle": True,
            "quality_gate": {"min_direction_margin": 0.03},
        },
    )
    assert resolved is not None
    direction, metrics = resolved
    assert direction == TradeDirection.CALL
    assert metrics.get("force_trade_every_cycle") is True
