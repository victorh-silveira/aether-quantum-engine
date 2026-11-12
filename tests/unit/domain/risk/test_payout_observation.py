"""Testes da atualizacao do payout cotado no risco da sessao."""

import pytest

from src.domain.risk.payout_observation import (
    apply_broker_payout_cap,
    cap_stake_for_max_payout,
    contract_profit_rate,
    parse_max_payout_rejection,
    record_observed_payout,
)


def test_contract_profit_rate_uses_gross_payout_minus_stake():
    assert contract_profit_rate(178.40, 100.00) == pytest.approx(0.784)
    assert contract_profit_rate(100.00, 100.00) is None
    assert contract_profit_rate(1.00, 0.00) is None
    assert contract_profit_rate(float("nan"), 10.0) is None
    assert contract_profit_rate(18.0, float("inf")) is None


def test_record_observed_payout_updates_only_after_valid_quote():
    params = {"payout_estimate": 0.85}
    assert record_observed_payout(params, payout=178.40, buy_price=100.00) == pytest.approx(0.784)
    assert params["payout_estimate"] == pytest.approx(0.784)
    assert params["observed_payout_rate"] == pytest.approx(0.784)

    assert record_observed_payout(params, payout=100.00, buy_price=100.00) is None
    assert params["payout_estimate"] == pytest.approx(0.784)


def test_parse_max_payout_rejection_variants():
    assert parse_max_payout_rejection("") is None
    assert parse_max_payout_rejection("Contract unavailable") is None
    assert parse_max_payout_rejection("Minimum stake of 0.50 and maximum payout of 0.00.") is None
    assert parse_max_payout_rejection(
        "Minimum stake of 0.50 and maximum payout of 100.00. Current payout is 161.52."
    ) == (100.0, 161.52)
    assert parse_max_payout_rejection("Maximum payout of 100.00 exceeded") == (100.0, 0.0)


def test_cap_stake_and_apply_broker_payout_cap():
    assert cap_stake_for_max_payout(0.0, 100.0, 150.0, 1.0) == 0.0
    assert cap_stake_for_max_payout(50.0, 0.0, 150.0, 1.0) == 0.0
    assert cap_stake_for_max_payout(50.0, 100.0, 150.0, 0.0) == 0.0
    assert cap_stake_for_max_payout(50.0, 100.0, 0.0, 1.0, is_rate=True) == 0.0
    assert cap_stake_for_max_payout(50.0, 100.0, 90.0, 1.0, is_rate=False) == 0.0
    assert cap_stake_for_max_payout(2.0, 1.0, 10.0, 1.0, is_rate=False) == 0.0

    scaled = cap_stake_for_max_payout(88.81, 100.0, 161.52, 1.0)
    assert scaled == pytest.approx(52.23)

    capped_rate = cap_stake_for_max_payout(88.81, 100.0, 0.7246, 1.0, is_rate=True)
    assert capped_rate == pytest.approx(55.66)

    eff, cov = apply_broker_payout_cap(88.81, 100.0, 0.7246, {"_broker_max_payout": 100.0}, 1.0)
    assert eff == pytest.approx(55.66)
    assert cov == pytest.approx(55.66 * 0.7246)

    eff_no_cap, cov_no_cap = apply_broker_payout_cap(10.0, 20.0, 0.8, None, 1.0)
    assert eff_no_cap == 10.0
    assert cov_no_cap == 8.0
