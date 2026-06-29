from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator.settlement_reconciliation import reconcile_after_ws_recovery


@pytest.mark.asyncio
async def test_reconcile_after_ws_recovery_settles_offline_contract():
    orch = MagicMock()
    orch._reconciliation_pending = False
    orch.config = {
        "orchestrator": {
            "execution": {
                "settlement_reconcile_timeout_seconds": 5.0,
                "settlement_reconcile_profit_table_limit": 50,
            }
        }
    }
    orch.state.active_contracts = {101: MagicMock()}
    orch.risk_manager.active_contract_ids = [101]
    orch.risk_manager.contract_to_symbol = {101: "R_10"}
    orch.ws = AsyncMock()
    orch.logger = MagicMock()
    orch._save_full_state = AsyncMock()
    orch.state.finalize_contract = AsyncMock(return_value=MagicMock())

    with (
        patch(
            "src.application.services.orchestrator.settlement_reconciliation.fetch_portfolio",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "src.application.services.orchestrator.settlement_reconciliation.reconcile_single_contract",
            new=AsyncMock(return_value=True),
        ),
        patch(
            "src.application.services.orchestrator.settlement_reconciliation.fetch_profit_table",
            new=AsyncMock(return_value=[]),
        ),
    ):
        result = await reconcile_after_ws_recovery(orch)

    assert result.settled_count == 1
    assert orch._reconciliation_pending is False
    orch._save_full_state.assert_awaited_once()


@pytest.mark.asyncio
async def test_reconcile_after_ws_recovery_late_settlement():
    orch = MagicMock()
    orch._reconciliation_pending = False
    orch.config = {
        "orchestrator": {
            "execution": {
                "settlement_reconcile_timeout_seconds": 5.0,
                "settlement_reconcile_profit_table_limit": 50,
            }
        }
    }
    orch.state.active_contracts = {}
    orch.risk_manager.active_contract_ids = []
    orch.risk_manager.contract_to_symbol = {202: "R_25"}
    orch.ws = AsyncMock()
    orch.logger = MagicMock()
    orch._save_full_state = AsyncMock()
    orch.state.finalize_contract = AsyncMock(return_value=None)

    row = {
        "contract_id": 202,
        "profit": -1.0,
        "contract_status": "lost",
        "symbol": "R_25",
    }
    with (
        patch(
            "src.application.services.orchestrator.settlement_reconciliation.fetch_portfolio",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "src.application.services.orchestrator.settlement_reconciliation.reconcile_single_contract",
            new=AsyncMock(return_value=False),
        ),
        patch(
            "src.application.services.orchestrator.settlement_reconciliation.fetch_profit_table",
            new=AsyncMock(return_value=[row]),
        ),
        patch(
            "src.application.services.orchestrator.settlement_reconciliation.process_late_settlement_from_payload",
            new=AsyncMock(),
        ) as late_mock,
    ):
        result = await reconcile_after_ws_recovery(orch)

    assert result.settled_count == 1
    late_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_reconcile_after_ws_recovery_processes_profit_row_with_finalize():
    orch = MagicMock()
    orch._reconciliation_pending = False
    orch.config = {
        "orchestrator": {
            "execution": {
                "settlement_reconcile_timeout_seconds": 5.0,
                "settlement_reconcile_profit_table_limit": 50,
            }
        }
    }
    orch.state.active_contracts = {}
    orch.risk_manager.active_contract_ids = []
    orch.risk_manager.contract_to_symbol = {303: "R_10"}
    orch.ws = AsyncMock()
    orch.logger = MagicMock()
    orch._save_full_state = AsyncMock()
    orch.state.finalize_contract = AsyncMock(return_value=MagicMock())
    row = {"contract_id": 303, "profit": 1.0, "contract_status": "won", "symbol": "R_10"}
    with (
        patch(
            "src.application.services.orchestrator.settlement_reconciliation.fetch_portfolio",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "src.application.services.orchestrator.settlement_reconciliation.reconcile_single_contract",
            new=AsyncMock(return_value=False),
        ),
        patch(
            "src.application.services.orchestrator.settlement_reconciliation.fetch_profit_table",
            new=AsyncMock(return_value=[row]),
        ),
        patch(
            "src.application.services.orchestrator.settlement_reconciliation.process_contract_settlement",
            new=AsyncMock(),
        ) as settle_mock,
    ):
        result = await reconcile_after_ws_recovery(orch)
    assert result.settled_count == 1
    settle_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_reconcile_skips_open_portfolio_contracts():
    orch = MagicMock()
    orch._reconciliation_pending = False
    orch.config = {"orchestrator": {"execution": {}}}
    orch.state.active_contracts = {505: MagicMock()}
    orch.risk_manager.active_contract_ids = [505]
    orch.risk_manager.contract_to_symbol = {}
    orch.ws = AsyncMock()
    orch.logger = MagicMock()
    orch._save_full_state = AsyncMock()
    with (
        patch(
            "src.application.services.orchestrator.settlement_reconciliation.fetch_portfolio",
            new=AsyncMock(return_value=[{"contract_id": 505}]),
        ),
        patch(
            "src.application.services.orchestrator.settlement_reconciliation.reconcile_single_contract",
            new=AsyncMock(),
        ) as reconcile_mock,
    ):
        result = await reconcile_after_ws_recovery(orch)
    reconcile_mock.assert_not_awaited()
    assert result.settled_count == 0
