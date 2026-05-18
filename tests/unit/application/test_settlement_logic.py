"""Testes unitários para o módulo settlement_logic."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator.settlement_logic import log_cluster_summary, process_contract_settlement


@pytest.mark.asyncio
async def test_process_contract_settlement_won():
    """Verifica o processamento de um contrato vencedor."""
    orch = MagicMock()
    orch.state = AsyncMock()
    orch.risk_manager = MagicMock()
    orch.running = True
    orch._contract_cycle = {123: 1}
    orch._buffer_result_logs = False
    orch._cluster_results = []
    orch._session_wins = 0
    orch._session_losses = 0

    data = {
        "proposal_open_contract": {
            "status": "won",
            "is_settled": 1,
            "contract_id": 123,
            "profit": 10.0,
            "balance_after": 1010.0,
        }
    }

    orch.state.finalize_contract.return_value = MagicMock()
    orch.state.balance = 1000.0

    with patch("src.application.services.orchestrator.settlement_logic.api_settlement_label", return_value="WIN"):
        await process_contract_settlement(orch, data)

    assert orch.state.balance == 1010.0
    assert orch._session_wins == 1
    orch.risk_manager.register_result.assert_called_once()


@pytest.mark.asyncio
async def test_process_contract_settlement_lost():
    """Verifica o processamento de um contrato perdedor."""
    orch = MagicMock()
    orch.state = AsyncMock()
    orch.risk_manager = MagicMock()
    orch.running = True
    orch._contract_cycle = {456: 2}
    orch._buffer_result_logs = True
    orch._pending_result_logs = []
    orch._cluster_results = []
    orch._session_wins = 0
    orch._session_losses = 0

    data = {
        "proposal_open_contract": {
            "status": "lost",
            "is_settled": 1,
            "contract_id": 456,
            "profit": -5.0,
            "balance_after": 995.0,
        }
    }

    orch.state.finalize_contract.return_value = MagicMock()
    orch.state.balance = 1000.0

    with patch("src.application.services.orchestrator.settlement_logic.api_settlement_label", return_value="LOSS"):
        await process_contract_settlement(orch, data)

    assert orch.state.balance == 995.0
    assert orch._session_losses == 1
    assert len(orch._pending_result_logs) == 1


def test_log_cluster_summary():
    """Verifica a emissão do log de resumo do cluster."""
    orch = MagicMock()
    orch._last_result_cycle_id = 1
    orch.state.balance = 1050.0
    orch._session_wins = 2
    orch._session_losses = 1
    orch._cluster_results = [{"some": "data"}]

    log_cluster_summary(orch)

    assert orch._cluster_results == []
    orch.logger.debug.assert_called()


@pytest.mark.asyncio
async def test_process_contract_settlement_early_exits():
    """Verifica saídas antecipadas em caso de dados inválidos."""
    orch = MagicMock()
    orch.state = AsyncMock()

    await process_contract_settlement(orch, {})
    orch.state.finalize_contract.assert_not_called()

    await process_contract_settlement(orch, {"proposal_open_contract": {"is_settled": 1}})
    orch.state.finalize_contract.assert_not_called()

    orch.state.finalize_contract.return_value = None
    await process_contract_settlement(orch, {"proposal_open_contract": {"is_settled": 1, "contract_id": 999}})
    orch.risk_manager.register_result.assert_not_called()


@pytest.mark.asyncio
async def test_process_contract_settlement_stop_win():
    """Verifica se o Stop Win desliga o orquestrador."""
    orch = MagicMock()
    orch.state = AsyncMock()
    orch.risk_manager = MagicMock()
    orch.running = True
    orch._contract_cycle = {777: 1}
    orch._buffer_result_logs = False
    orch._cluster_results = []
    orch._session_wins = 0
    orch._session_losses = 0
    orch.config = {"risk_management": {"stop_win_percentage": 5.0}}

    data = {
        "proposal_open_contract": {
            "status": "won",
            "is_settled": 1,
            "contract_id": 777,
            "profit": 100.0,
            "balance_after": 1100.0,
        }
    }

    orch.state.finalize_contract.return_value = MagicMock()
    orch.state.balance = 1000.0
    orch.risk_manager.initial_bankroll = 1000.0
    orch.risk_manager.total_session_profit = 100.0
    orch.risk_manager.active_contract_ids = []  # Simula fim do cluster

    with (
        patch("src.application.services.orchestrator.settlement_logic.resolve_stop_win_target", return_value=50.0),
        patch("src.application.services.orchestrator.settlement_logic.api_settlement_label", return_value="WIN"),
    ):
        await process_contract_settlement(orch, data)

    assert orch.running is False
    orch.logger.debug.assert_any_call("[C%04d] STOP_WIN | pnl_sessao=$%+.2f | alvo=$%.2f", 1, 100.0, 50.0)
