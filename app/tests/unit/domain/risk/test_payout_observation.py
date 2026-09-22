"""Testes da atualizacao do payout cotado no risco da sessao."""

import pytest

from src.domain.risk.payout_observation import contract_profit_rate, record_observed_payout


def test_contract_profit_rate_uses_gross_payout_minus_stake():
    assert contract_profit_rate(178.40, 100.00) == pytest.approx(0.784)
    assert contract_profit_rate(100.00, 100.00) is None
    assert contract_profit_rate(1.00, 0.00) is None


def test_record_observed_payout_updates_only_after_valid_quote():
    params = {"payout_estimate": 0.85}
    assert record_observed_payout(params, payout=178.40, buy_price=100.00) == pytest.approx(0.784)
    assert params["payout_estimate"] == pytest.approx(0.784)
    assert params["observed_payout_rate"] == pytest.approx(0.784)

    assert record_observed_payout(params, payout=100.00, buy_price=100.00) is None
    assert params["payout_estimate"] == pytest.approx(0.784)
