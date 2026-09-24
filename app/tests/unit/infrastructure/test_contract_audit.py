"""Auditoria preserva valores do broker e distingue inferencias."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.application.services.orchestrator.settlement_audit import record_settlement_audit
from src.infrastructure.market.contract_audit import open_audit_row, settlement_audit_row


def test_open_audit_keeps_unknown_spots_null():
    row = open_audit_row(
        {"contract_id": "7", "transaction_id": 3, "start_time": "100", "buy_price": "10", "payout": "19.5"},
        symbol="1HZ75V",
        mode="demo",
        direction="CALL",
        request_epoch_ms=1,
        ack_epoch_ms=2,
    )
    assert row["date_start"] == 100
    assert row["date_expiry"] is None
    assert row["transaction_buy_id"] == "3"
    assert row["settlement_source"] == "pending"
    assert "entry_tick" not in row
    assert row["payout"] == 19.5


def test_settlement_audit_prefers_current_spot_keys_and_handles_missing():
    row = settlement_audit_row(
        {
            "contract_id": 7,
            "entry_spot": "100.5",
            "entry_tick": "100.0",
            "entry_spot_time": 10,
            "exit_spot": "101.0",
            "exit_spot_time": 310,
            "profit": "9.5",
            "status": "won",
        },
        symbol="1HZ75V",
        mode="real",
        direction="CALL",
        signal_prob=0.61,
    )
    assert (row["entry_tick"], row["exit_tick"]) == (100.5, 101.0)
    assert (row["entry_tick_time"], row["exit_tick_time"]) == (10, 310)
    assert row["settlement_source"] == "broker"
    assert row["signal_prob"] == 0.61
    inferred = settlement_audit_row(
        {"contract_id": 8, "audit_source": "inferred_rest", "status": "lost", "profit": -10},
        symbol="1HZ75V",
        mode="demo",
        direction="PUT",
    )
    assert inferred["entry_tick"] is None
    assert inferred["settlement_source"] == "inferred_rest"
    assert inferred["signal_prob"] is None
    unknown = settlement_audit_row(
        {"contract_id": 9, "audit_source": "untrusted", "entry_tick": 1, "exit_tick": 2},
        symbol="1HZ75V",
        mode="demo",
        direction="CALL",
    )
    assert unknown["settlement_source"] == "unknown"


@pytest.mark.asyncio
async def test_record_settlement_audit_best_effort(caplog):
    orch = MagicMock()
    orch.config = {"trading": {"mode": "real"}}
    orch.market_writer.enqueue_contract_audit = AsyncMock()
    contract = MagicMock()
    contract.direction.value = "PUT"
    payload = {"contract_id": 7, "entry_spot": 1.0, "exit_spot": 0.9}
    await record_settlement_audit(orch, payload, contract, "1HZ75V", signal_prob=0.4)
    row = orch.market_writer.enqueue_contract_audit.call_args.args[0]
    assert (row["account_mode"], row["direction"]) == ("real", "PUT")
    assert row["signal_prob"] == 0.4
    orch.market_writer.enqueue_contract_audit.side_effect = RuntimeError("db")
    await record_settlement_audit(orch, payload, contract, "1HZ75V")
    assert "sem captura" in caplog.text
    orch.market_writer = None
    await record_settlement_audit(orch, payload, contract, "1HZ75V")
    orch.market_writer = object()
    await record_settlement_audit(orch, payload, contract, "1HZ75V")
