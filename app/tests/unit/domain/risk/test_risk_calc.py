import pytest

from src.domain.risk.executed_stake_reconciliation import (
    bind_executed_stake_for_contract,
    reconcile_settlement_profit,
    resolve_executed_buy_stake,
)
from src.domain.risk.risk_manager import RiskManager


def test_resolve_executed_buy_stake_prefers_api_payload():
    stake = resolve_executed_buy_stake(
        42,
        payload={"buy_price": 332.28},
        contract_stakes={42: 390.92},
    )
    assert stake == pytest.approx(332.28)


def test_resolve_executed_buy_stake_falls_back_to_contract_buy_price():
    contract = type("C", (), {"buy_price": 332.28, "stake": 390.92})()
    stake = resolve_executed_buy_stake(7, contract=contract, contract_stakes={7: 390.92})
    assert stake == pytest.approx(332.28)


def test_reconcile_settlement_profit_loss_uses_executed_buy_not_planned():
    assert reconcile_settlement_profit(-390.92, 332.28) == pytest.approx(-332.28)


def test_reconcile_settlement_profit_keeps_positive_api_profit():
    assert reconcile_settlement_profit(85.0, 332.28) == pytest.approx(85.0)


def test_bind_executed_stake_overwrites_planned_contract_stake():
    stakes: dict[int, float] = {99: 390.92}
    bind_executed_stake_for_contract(stakes, 99, 332.28)
    assert stakes[99] == pytest.approx(332.28)


def test_resolve_executed_buy_stake_uses_purchase_price_and_amount_keys():
    assert resolve_executed_buy_stake(1, payload={"purchase_price": 250.0}) == pytest.approx(250.0)
    assert resolve_executed_buy_stake(2, payload={"amount": 180.5}) == pytest.approx(180.5)


def test_resolve_executed_buy_stake_falls_back_to_contract_stake():
    contract = type("C", (), {"buy_price": None, "stake": 390.92})()
    stake = resolve_executed_buy_stake(8, contract=contract)
    assert stake == pytest.approx(390.92)


def test_resolve_executed_buy_stake_falls_back_to_recorded_contract_stakes():
    stake = resolve_executed_buy_stake(11, contract_stakes={11: 332.28})
    assert stake == pytest.approx(332.28)


def test_resolve_executed_buy_stake_returns_zero_when_unresolved():
    assert resolve_executed_buy_stake(0) == 0.0


def test_bind_executed_stake_skips_non_positive_values():
    stakes: dict[int, float] = {77: 390.92}
    bind_executed_stake_for_contract(stakes, 77, 0.0)
    assert stakes[77] == pytest.approx(390.92)


def test_reconcile_settlement_profit_keeps_api_loss_when_executed_buy_missing():
    assert reconcile_settlement_profit(-50.0, 0.0) == pytest.approx(-50.0)


def test_register_result_pending_uses_executed_stake_after_reconciliation(kelly_config):
    rm = RiskManager(kelly_config)
    rm.active_contract_ids = [501]
    rm.contract_to_symbol[501] = "RDBULL"
    rm.contract_stakes[501] = 390.92
    bind_executed_stake_for_contract(rm.contract_stakes, 501, 332.28)
    rm.register_result(-332.28, 501, symbol="RDBULL")
    assert rm.pending_loss["RDBULL"] == pytest.approx(332.28)
    assert rm.last_loss_stake == pytest.approx(332.28)
    assert rm.total_session_profit == pytest.approx(-332.28)
