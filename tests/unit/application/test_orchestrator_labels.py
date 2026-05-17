"""Testes para labels de liquidação do Orchestrator."""

from src.application.services.orchestrator import Orchestrator


def test_api_settlement_label_maps_status_and_profit():
    cases = [
        ("won", 0.0, "WIN"),
        ("lost", -5.0, "LOSS"),
        ("", 3.0, "WIN"),
        ("", -2.0, "LOSS"),
        ("", 0.0, "FLAT"),
        ("expired", 1.0, "WIN"),
        ("expired", -1.0, "LOSS"),
        ("expired", 0.0, "FLAT"),
    ]
    for st, pr, res in cases:
        assert Orchestrator._api_settlement_label(st, pr) == res
