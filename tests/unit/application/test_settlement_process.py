"""Testes unitários para o módulo settlement_process."""

from unittest.mock import MagicMock, patch

from src.application.services.orchestrator.settlement_process import (
    emit_contract_settlement_log,
    process_settlement_outcome,
)


def test_emit_contract_settlement_log_direct_and_fallbacks():
    orch = MagicMock()
    orch._buffer_result_logs = False
    orch._contract_cycle = {}
    orch._last_result_cycle_id = 7
    orch.risk_manager.contract_to_symbol = {}
    mock_contract = MagicMock()
    mock_contract.symbol = "1HZ25V"
    orch.state.active_contracts = {123: mock_contract}

    emit_contract_settlement_log(orch, "RESOLVED WIN", c_id=123, sym=None, cycle_id=None)
    orch.logger.info.assert_called_once_with("RESOLVED WIN")


def test_emit_contract_settlement_log_buffered():
    orch = MagicMock()
    orch._buffer_result_logs = True
    orch._pending_result_logs = []
    orch._contract_cycle = {456: 3}
    orch.risk_manager.contract_to_symbol = {456: "R_25"}

    emit_contract_settlement_log(orch, "RESOLVED LOSS", c_id=456, sym="R_25", cycle_id=3)
    assert len(orch._pending_result_logs) == 1
    assert "RESOLVED LOSS" in orch._pending_result_logs[0]


def test_process_settlement_outcome_delegates():
    orch = MagicMock()
    c = {"contract_id": 100}
    with patch("src.application.services.orchestrator.settlement_process.process_contract_outcome") as mock_outcome:
        process_settlement_outcome(orch, c, None, 100, 10.5, audit_direction="CALL", audit_raw_prob=0.6)
        mock_outcome.assert_called_once()
