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
