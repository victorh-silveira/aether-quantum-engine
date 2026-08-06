"""Pass-through de direcao TCN sem vetos de sinal."""

from src.application.services.execution_direction_checks import (
    initial_direction_checks,
    is_technically_blocked,
)
from src.application.services.execution_direction_resolver import resolve_execution_direction
from src.domain.models.trade import TradeDirection


def test_initial_direction_checks_allows_hurst_mid_band():
    entry = {
        "metrics": {
            "calibrated_prob": 0.62,
            "deploy_ok": True,
            "indicators": {"hurst": 0.50, "adx": 0.05},
        }
    }
    result = initial_direction_checks(entry, {})
    assert result is not None
    dl_dir, metrics, prob = result
    assert dl_dir == TradeDirection.CALL
    assert prob == 0.62
    assert metrics["resolved_direction"] == "CALL"


def test_is_technically_blocked_deploy_and_training():
    assert is_technically_blocked({"metrics": {"deploy_ok": False}}) is True
    assert is_technically_blocked({"metrics": {"gate_reason": "training"}}) is True
    assert is_technically_blocked({"metrics": {"deploy_ok": True, "calibrated_prob": 0.7}}) is False


def test_resolve_execution_direction_pass_through_negative_edge():
    entry = {
        "metrics": {
            "calibrated_prob": 0.58,
            "deploy_ok": True,
            "predicted_payoff_edge": -0.20,
            "indicators": {"hurst": 0.50, "adx": 0.10},
        }
    }
    result = resolve_execution_direction(entry, exec_cfg={}, symbol="OTC_SPC")
    assert result is not None
    direction, metrics = result
    assert direction in {TradeDirection.CALL, TradeDirection.PUT}
    assert metrics.get("execution_candidate_ready") is True
    assert metrics.get("gate_reason") is None


def test_resolve_execution_direction_blocks_technical():
    entry = {"metrics": {"calibrated_prob": 0.70, "deploy_ok": False, "gate_reason": "deploy"}}
    assert resolve_execution_direction(entry, exec_cfg={}) is None
