"""Atribuicao de direcao pelo resultado confirmado do contrato."""

from src.application.services.market_audit_contract import broker_price_audit


def test_broker_price_audit_reports_real_direction_and_phase():
    row = {"entry_spot": 4812.45, "exit_spot": 4812.78, "entry_spot_time": 1791167104}
    assert broker_price_audit(row, "PUT") == "SPOT: 4812.45->4812.78 (+0.33) | REAL: CALL | FASE_M5: 4s"


def test_broker_price_audit_flat_and_missing_phase():
    assert broker_price_audit({"entry_tick": 100, "exit_tick": 100}, "CALL") == (
        "SPOT: 100.00->100.00 (+0.00) | REAL: FLAT"
    )
    assert broker_price_audit({"entry_spot": 101, "exit_spot": 100, "entry_spot_time": "bad"}, "PUT") == (
        "SPOT: 101.00->100.00 (-1.00) | REAL: PUT"
    )


def test_broker_price_audit_rejects_unconfirmed_or_invalid_data():
    assert broker_price_audit({"audit_source": "inferred_rest", "entry_spot": 1, "exit_spot": 2}, "CALL") is None
    assert broker_price_audit({"entry_spot": None, "exit_spot": 2}, "CALL") is None
    assert broker_price_audit({"entry_spot": "bad", "exit_spot": 2}, "CALL") is None
    assert broker_price_audit({"entry_spot": float("nan"), "exit_spot": 2}, "CALL") is None
    assert broker_price_audit({"entry_spot": -1, "exit_spot": 2}, "CALL") is None
    assert broker_price_audit({"entry_spot": 1, "exit_spot": 2}, "UNKNOWN") is None
