"""Testes de restore_state no RiskManager."""

from src.domain.risk.risk_manager import RiskManager


def test_risk_manager_restore_state(kelly_config):
    rm = RiskManager(kelly_config)
    snapshot = rm.get_state()
    snapshot["consecutive_losses"] = 4
    snapshot["pending_loss"] = {"R_25": 3.5}
    snapshot["rolling_wins"] = {"R_10": [1, 0, 1]}
    rm.restore_state({})
    rm.restore_state(snapshot)
    assert rm.consecutive_losses == 4
    assert rm.pending_loss["R_25"] == 3.5
    assert rm._rolling_wins["R_10"] == [1, 0, 1]
