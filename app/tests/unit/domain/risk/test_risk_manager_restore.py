"""Testes de restore_state no RiskManager."""

from src.domain.risk.risk_manager import RiskManager


def test_risk_manager_restore_state(kelly_config):
    rm = RiskManager(kelly_config)
    snapshot = rm.get_state()
    snapshot["consecutive_losses_linear"] = 4
    snapshot["dlambert_unit"] = 22.5
    snapshot["pending_loss"] = {"R_10": 3.5}
    snapshot["rolling_wins"] = {"R_10": [1, 0, 1]}
    rm.restore_state({})
    rm.restore_state(snapshot)
    assert rm.consecutive_losses_linear == 4
    assert rm.dlambert_unit == 22.5
    assert rm.pending_loss["R_10"] == 3.5
    assert rm._rolling_wins["R_10"] == [1, 0, 1]


def test_risk_manager_restore_legacy_consecutive_losses(kelly_config):
    rm = RiskManager(kelly_config)
    rm.restore_state({"consecutive_losses": 2, "dlambert_unit": 10.0})
    assert rm.consecutive_losses_linear == 2


def test_risk_manager_restore_ignores_last_martingale_stake(kelly_config):
    rm = RiskManager(kelly_config)
    rm.restore_state({"last_martingale_stake": 999.0, "dlambert_unit": 12.0})
    assert rm.dlambert_unit == 12.0
    assert not hasattr(rm, "last_martingale_stake")
