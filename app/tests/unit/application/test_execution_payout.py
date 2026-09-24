"""Testes do payout usado pelo edge no ciclo de execucao."""

from types import SimpleNamespace

import pytest

from src.application.services.execution_payout import resolve_execution_payout


def test_resolve_execution_payout_prefers_observed_session_quote():
    orch = SimpleNamespace(risk_manager=SimpleNamespace(risk_params={"observed_payout_rate": 0.7838}))
    assert resolve_execution_payout(orch) == pytest.approx(0.7838)


def test_resolve_execution_payout_falls_back_to_configured_rate():
    assert resolve_execution_payout(None) == pytest.approx(0.85)
    orch = SimpleNamespace(risk_manager=SimpleNamespace(risk_params={"payout_estimate": "invalid"}))
    assert resolve_execution_payout(orch) == pytest.approx(0.85)


def test_resolve_execution_payout_uses_trade_handler_latest_rate():
    orch = SimpleNamespace(
        risk_manager=None,
        trade_handler=SimpleNamespace(latest_payout_rate=0.791),
    )
    assert resolve_execution_payout(orch) == pytest.approx(0.791)
    orch_invalid = SimpleNamespace(
        risk_manager=None,
        trade_handler=SimpleNamespace(latest_payout_rate="invalid"),
    )
    assert resolve_execution_payout(orch_invalid) == pytest.approx(0.85)
