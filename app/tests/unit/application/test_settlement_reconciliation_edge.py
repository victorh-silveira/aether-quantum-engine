from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator.settlement_reconciliation import (
    _known_contract_ids,
    _profit_table_limit,
    _reconcile_timeout,
    reconcile_after_ws_recovery,
)


@pytest.mark.asyncio
async def test_reconcile_transient_error_marks_ws_offline():
    orch = MagicMock()
    orch._reconciliation_pending = False
    orch.config = {"orchestrator": {"execution": {}}}
    orch.state.active_contracts = {808: MagicMock()}
    orch.risk_manager.active_contract_ids = [808]
    orch.risk_manager.contract_to_symbol = {}
    orch.ws = MagicMock()
    orch.ws.is_running = True
    orch.logger = MagicMock()
    orch._save_full_state = AsyncMock()
    with (
        patch(
            "src.application.services.orchestrator.settlement_reconciliation.fetch_portfolio",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "src.application.services.orchestrator.settlement_reconciliation.reconcile_single_contract",
            new=AsyncMock(side_effect=ConnectionError("offline")),
        ),
    ):
        result = await reconcile_after_ws_recovery(orch)
    assert orch.ws.is_running is False
    assert any("ConnectionError" in err for err in result.errors)


@pytest.mark.asyncio
async def test_reconcile_outer_exception_marks_ws_offline():
    orch = MagicMock()
    orch._reconciliation_pending = False
    orch.config = {"orchestrator": {"execution": {}}}
    orch.ws = AsyncMock()
    orch.logger = MagicMock()
    orch._save_full_state = AsyncMock()
    with patch(
        "src.application.services.orchestrator.settlement_reconciliation.fetch_portfolio",
        new=AsyncMock(side_effect=ConnectionError("down")),
    ):
        result = await reconcile_after_ws_recovery(orch)
    assert "ConnectionError" in result.errors[0]

    orch = MagicMock()
    orch._reconciliation_pending = False
    orch.config = {"orchestrator": {"execution": {}}}
    orch.state.active_contracts = {404: MagicMock()}
    orch.risk_manager.active_contract_ids = [404]
    orch.risk_manager.contract_to_symbol = {}
    orch.ws = AsyncMock()
    orch.logger = MagicMock()
    orch._save_full_state = AsyncMock()
    with (
        patch(
            "src.application.services.orchestrator.settlement_reconciliation.fetch_portfolio",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "src.application.services.orchestrator.settlement_reconciliation.reconcile_single_contract",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ),
    ):
        result = await reconcile_after_ws_recovery(orch)
    assert result.errors


def test_reconcile_timeout_and_profit_limit_defaults():
    assert _reconcile_timeout(MagicMock(config={"orchestrator": {"execution": "bad"}})) == 45.0
    assert _profit_table_limit(MagicMock(config={"orchestrator": {"execution": None}})) == 120


def test_known_contract_ids_skips_invalid_entries():
    orch = MagicMock()
    orch.state.active_contracts = {"bad": MagicMock(), 10: MagicMock()}
    orch.risk_manager.active_contract_ids = ["x", 11]
    orch.risk_manager.contract_to_symbol = {None: "R_10", 12: "R_10"}
    assert _known_contract_ids(orch) == [10, 11, 12]


@pytest.mark.asyncio
async def test_reconcile_profit_burst_skips_missing_row():
    orch = MagicMock()
    orch._reconciliation_pending = False
    orch.config = {"orchestrator": {"execution": {}}}
    orch.state.active_contracts = {}
    orch.risk_manager.active_contract_ids = []
    orch.risk_manager.contract_to_symbol = {601: "R_10"}
    orch.ws = AsyncMock()
    orch.logger = MagicMock()
    orch._save_full_state = AsyncMock()
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
            new=AsyncMock(return_value=[{"contract_id": 999, "profit": 1.0}]),
        ),
    ):
        result = await reconcile_after_ws_recovery(orch)
    assert result.settled_count == 0


@pytest.mark.asyncio
async def test_reconcile_profit_burst_skips_bad_payload():
    orch = MagicMock()
    orch._reconciliation_pending = False
    orch.config = {"orchestrator": {"execution": {}}}
    orch.state.active_contracts = {}
    orch.risk_manager.active_contract_ids = []
    orch.risk_manager.contract_to_symbol = {602: "R_10"}
    orch.ws = AsyncMock()
    orch.logger = MagicMock()
    orch._save_full_state = AsyncMock()
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
            new=AsyncMock(return_value=[{"contract_id": 602, "profit": 1.0}]),
        ),
        patch(
            "src.application.services.orchestrator.settlement_reconciliation.settlement_payload_from_profit_row",
            return_value={"proposal_open_contract": "bad"},
        ),
    ):
        result = await reconcile_after_ws_recovery(orch)
    assert result.settled_count == 0


@pytest.mark.asyncio
async def test_reconcile_profit_burst_late_settlement_path():
    orch = MagicMock()
    orch._reconciliation_pending = False
    orch.config = {"orchestrator": {"execution": {}}}
    orch.state.active_contracts = {}
    orch.risk_manager.active_contract_ids = []
    orch.risk_manager.contract_to_symbol = {602: "R_10"}
    orch.ws = AsyncMock()
    orch.logger = MagicMock()
    orch._save_full_state = AsyncMock()
    orch.state.finalize_contract = AsyncMock(return_value=None)
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
            new=AsyncMock(return_value=[{"contract_id": 602, "profit": 1.0}]),
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
async def test_reconcile_profit_burst_exception_recorded():
    orch = MagicMock()
    orch._reconciliation_pending = False
    orch.config = {"orchestrator": {"execution": {}}}
    orch.state.active_contracts = {}
    orch.risk_manager.active_contract_ids = []
    orch.risk_manager.contract_to_symbol = {701: "R_10"}
    orch.ws = AsyncMock()
    orch.logger = MagicMock()
    orch._save_full_state = AsyncMock()
    orch.state.finalize_contract = AsyncMock(return_value=None)
    row = {"contract_id": 701, "profit": -1.0, "contract_status": "lost"}
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
            new=AsyncMock(side_effect=RuntimeError("late-fail")),
        ),
    ):
        result = await reconcile_after_ws_recovery(orch)
    assert any("RuntimeError" in err for err in result.errors)
