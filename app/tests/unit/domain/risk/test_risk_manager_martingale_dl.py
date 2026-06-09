import math

import pytest

from src.domain.risk.risk_manager import RiskManager


def test_martingale_uses_recorded_loss_stake_as_seed(kelly_config):
    rm = RiskManager(kelly_config)
    rm.active_contract_ids = [1]
    rm.record_contract_stake(1, 10.83)
    rm.register_result(-10.83, 1, "R_100")
    stake = rm.calculate_stake(10820.0, "R_10", conviction=0.0)
    cover = (10.83 + 10.83 * 0.95) / 0.95
    assert stake == pytest.approx(math.ceil(cover * 100) / 100, abs=0.5)


def test_martingale_always_on_with_pending(kelly_config):
    rm = RiskManager(kelly_config)
    rm.pending_loss["R_50"] = 50.0
    assert rm._martingale_allowed(
        "R_50",
        0.40,
        dl_metrics={"gate_reason": "deploy", "val_brier": 0.35, "deploy_ok": False},
        order_direction="PUT",
    )
    assert rm._martingale_allowed(
        "R_75",
        0.56,
        dl_metrics={"gate_reason": "brier", "val_brier": 0.40},
        order_direction="CALL",
        last_loss_symbol="R_50",
        last_loss_direction="CALL",
    )


def test_martingale_stake_grows_with_pending_loss(kelly_config):
    rm = RiskManager(kelly_config)
    rm.active_contract_ids = [1]
    rm.register_result(-10.0, 1, "R_50")
    first = rm.calculate_stake(1000.0, "R_75", conviction=0.56)
    rm.active_contract_ids = [2]
    rm.register_result(-20.0, 2, "R_75")
    second = rm.calculate_stake(970.0, "R_75", conviction=0.56)
    assert second > first


def test_symbol_loss_cooldown_records_direction(kelly_config):
    kelly_config["kelly"]["symbol_loss_cooldown_cycles"] = 1
    rm = RiskManager(kelly_config)
    rm.active_contract_ids = [1]
    rm.register_result(-5.0, 1, "R_50", direction="PUT")
    assert rm.last_loss_direction == "PUT"
