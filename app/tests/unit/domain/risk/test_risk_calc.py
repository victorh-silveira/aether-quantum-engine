import pytest

from src.domain.risk.executed_stake_reconciliation import (
    apply_contract_settlement_result,
    apply_fractional_payoff_residual_to_pending,
    bind_executed_stake_for_contract,
    domain_expected_win_profit,
    fractional_payoff_residual_cents,
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
    rm.contract_to_symbol[501] = "R_10"
    rm.contract_stakes[501] = 390.92
    bind_executed_stake_for_contract(rm.contract_stakes, 501, 332.28)
    rm.register_result(-332.28, 501, symbol="R_10")
    assert rm.pending_loss["R_10"] == pytest.approx(332.28)
    assert rm.last_loss_stake == pytest.approx(332.28)
    assert rm.total_session_profit == pytest.approx(-332.28)


def test_domain_expected_win_profit_uses_stake_times_payout():
    assert domain_expected_win_profit(100.0, 0.95) == pytest.approx(95.0)


def test_fractional_payoff_residual_cents_detects_underpayment():
    residual = fractional_payoff_residual_cents(94.99, 100.0, 0.95)
    assert residual == pytest.approx(-0.01)


def test_fractional_payoff_residual_cents_detects_overpayment():
    residual = fractional_payoff_residual_cents(95.02, 100.0, 0.95)
    assert residual == pytest.approx(0.02)


def test_fractional_payoff_residual_cents_returns_zero_for_non_win_or_missing_stake():
    assert fractional_payoff_residual_cents(0.0, 100.0, 0.95) == 0.0
    assert fractional_payoff_residual_cents(10.0, 0.0, 0.95) == 0.0


def test_fractional_payoff_residual_cents_ignores_structural_mismatch():
    residual = fractional_payoff_residual_cents(85.0, 297.34, 0.95)
    assert residual == 0.0


def test_apply_fractional_payoff_residual_increases_pending_on_underpayment():
    pending = {"R_10": 10.0}
    apply_fractional_payoff_residual_to_pending(pending, "R_10", -0.02)
    assert pending["R_10"] == pytest.approx(10.02)


def test_apply_fractional_payoff_residual_clears_symbol_when_overpayment_exhausts_pending():
    pending = {"R_10": 0.02}
    apply_fractional_payoff_residual_to_pending(pending, "R_10", 0.03)
    assert "R_10" not in pending


def test_apply_fractional_payoff_residual_reduces_pending_on_overpayment():
    pending = {"R_10": 10.0}
    apply_fractional_payoff_residual_to_pending(pending, "R_10", 0.03)
    assert pending["R_10"] == pytest.approx(9.97)


def test_apply_contract_settlement_result_reconciles_fractional_win_residual(kelly_config):
    rm = RiskManager(kelly_config)
    rm.active_contract_ids = [901]
    rm.contract_to_symbol[901] = "R_10"
    rm.contract_stakes[901] = 100.0
    rm.pending_loss["R_10"] = 20.0
    rm.begin_cluster(1)
    apply_contract_settlement_result(rm, 94.99, 901, "R_10")
    assert "R_10" not in rm.pending_loss
