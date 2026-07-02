import pytest

from src.domain.risk.consensus_stake_penalty import consensus_kelly_retention
from src.domain.risk.risk_manager import RiskManager
from src.domain.risk.risk_recovery_state import (
    apply_cluster_profit_to_recovery_state,
    apply_dlambert_partial_win_retraction,
    pending_loss_total,
    recovery_financially_active,
)


_CFG = {
    "consensus_penalty_enabled": True,
    "consensus_max_cut": 0.50,
    "consensus_di_weight": 0.30,
    "consensus_cmo_weight": 0.30,
    "consensus_rsi_weight": 0.25,
    "consensus_entropy_exponent": 2.0,
    "penalty_smoothing_factor": 0.40,
    "penalty_smoothing_trade_score_min": 0.70,
}


def test_pending_loss_total_and_recovery_flag():
    pending = {"RDBEAR": 4.0, "RDBULL": 2.5}
    assert pending_loss_total(pending) == pytest.approx(6.5)
    assert recovery_financially_active(pending) is True
    assert recovery_financially_active({}) is False


def test_apply_dlambert_partial_win_retraction_noop_at_zero():
    rm = type("RM", (), {"consecutive_losses_linear": 0})()
    apply_dlambert_partial_win_retraction(rm)
    assert rm.consecutive_losses_linear == 0


def test_cluster_win_retracts_linear_while_pending():
    rm = type("RM", (), {})()
    rm.pending_loss = {"RDBEAR": 12.0}
    rm.consecutive_losses_linear = 2
    rm.total_session_profit = -8.0
    rm.last_loss_stake = 10.0
    rm.logger = type("L", (), {"info": lambda *a, **k: None})()

    apply_cluster_profit_to_recovery_state(rm, 3.0)

    assert rm.consecutive_losses_linear == 1
    assert rm.last_loss_stake == 10.0


def test_cluster_win_resets_when_pending_cleared():
    rm = type("RM", (), {})()
    rm.pending_loss = {}
    rm.consecutive_losses_linear = 2
    rm.total_session_profit = 4.0
    rm.last_loss_stake = 10.0
    rm.logger = type("L", (), {"info": lambda *a, **k: None})()

    apply_cluster_profit_to_recovery_state(rm, 5.0)

    assert rm.consecutive_losses_linear == 0
    assert rm.last_loss_stake == 0.0


def test_recovery_penalty_waived_with_stable_trade_score():
    metrics = {
        "call_votes": 1,
        "put_votes": 5,
        "trade_score": 0.72,
        "indicators": {"di_diff": 0.01, "cmo": -0.18, "rsi": 0.36},
    }
    raw = consensus_kelly_retention(metrics, "CALL", kelly_config=_CFG)
    assert raw < 1.0
    metrics_copy = dict(metrics)
    waived = consensus_kelly_retention(
        metrics_copy,
        "CALL",
        kelly_config=_CFG,
        consecutive_losses=2,
        pending_loss_total=18.0,
    )
    assert waived == 1.0
    assert metrics_copy.get("consensus_penalty_recovery_waived") is True


def test_recovery_penalty_smoothing_skips_low_trade_score():
    metrics = {
        "call_votes": 1,
        "put_votes": 5,
        "trade_score": 0.62,
        "indicators": {"di_diff": 0.01, "cmo": -0.18, "rsi": 0.36},
    }
    base = consensus_kelly_retention(metrics, "CALL", kelly_config=_CFG)
    with_recovery = consensus_kelly_retention(
        dict(metrics),
        "CALL",
        kelly_config=_CFG,
        consecutive_losses=2,
        pending_loss_total=18.0,
    )
    assert with_recovery == base


def test_consensus_kelly_retention_non_dict_metrics():
    assert consensus_kelly_retention(None, "CALL", kelly_config=_CFG) == 1.0


def test_cluster_loss_increments_linear_counter():
    rm = type("RM", (), {})()
    rm.pending_loss = {"RDBEAR": 5.0}
    rm.consecutive_losses_linear = 1
    rm.total_session_profit = -12.0
    rm.logger = type("L", (), {"info": lambda *a, **k: None})()

    apply_cluster_profit_to_recovery_state(rm, -4.0)

    assert rm.consecutive_losses_linear == 2


def test_linear_retraction_on_partial_win_with_pending(kelly_config):
    rm = RiskManager(kelly_config)
    rm.active_contract_ids = [1]
    rm.register_result(-10.0, 1, "RDBULL")
    assert rm.consecutive_losses_linear == 1
    assert rm.recovery_financially_active()

    rm.active_contract_ids = [2]
    rm.register_result(3.0, 2, "RDBULL")
    assert rm.consecutive_losses_linear == 1
    assert rm.pending_loss_total() == pytest.approx(7.0)

    rm.active_contract_ids = [3]
    rm.register_result(8.0, 3, "RDBULL")
    assert rm.consecutive_losses_linear == 0
    assert not rm.recovery_financially_active()
