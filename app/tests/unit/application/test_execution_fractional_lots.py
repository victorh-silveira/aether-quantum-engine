from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator import Orchestrator
from src.application.services.orchestrator.execution_fractional_lots import (
    MAX_SINGLE_STAKE_LIMIT,
    _stagger_fractional_dispatch,
    dispatch_fractional_orders,
    register_contract_lot_group,
    resolve_fractional_lot_stagger_seconds,
    resolve_max_single_stake_limit,
    split_fractional_stake_lots,
)
from src.domain.models.trade import TradeDirection


def test_split_fractional_stake_lots_splits_into_three_lots():
    lots = split_fractional_stake_lots(550.0, limit=200.0)
    assert len(lots) == 3
    assert sum(lots) == pytest.approx(550.0)
    assert lots[-1] == pytest.approx(550.0 - lots[0] - lots[1])


def test_resolve_max_single_stake_limit_reads_exec_cfg():
    assert resolve_max_single_stake_limit({"max_single_stake_limit": 180.0}) == pytest.approx(180.0)
    assert resolve_max_single_stake_limit(None) == pytest.approx(MAX_SINGLE_STAKE_LIMIT)


def test_resolve_fractional_lot_stagger_seconds_reads_orchestrator_config(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch.config.setdefault("orchestrator", {}).setdefault("execution", {})["fractional_lot_stagger_jitter_us"] = 0.0
        delay = resolve_fractional_lot_stagger_seconds(orch)
        assert delay >= 0.0


def test_register_contract_lot_group_initializes_missing_maps():
    orch = type("Orch", (), {"_active_cycle_id": 9})()
    group_id = register_contract_lot_group(orch, [2001])
    assert group_id == 9
    assert orch._contract_lot_groups[9] == (2001,)
    assert orch._contract_lot_group[2001] == 9


@pytest.mark.asyncio
async def test_stagger_fractional_dispatch_sleeps_for_followup_lot(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        with patch(
            "src.application.services.orchestrator.execution_fractional_lots.asyncio.sleep",
            new_callable=AsyncMock,
        ) as sleep_mock:
            await _stagger_fractional_dispatch(orch, 1)
        sleep_mock.assert_awaited_once()


def test_split_fractional_stake_lots_keeps_small_stake_intact():
    assert split_fractional_stake_lots(150.0, limit=MAX_SINGLE_STAKE_LIMIT) == [150.0]


def test_split_fractional_stake_lots_splits_large_stake_evenly():
    lots = split_fractional_stake_lots(268.82, limit=MAX_SINGLE_STAKE_LIMIT)
    assert len(lots) == 2
    assert lots[0] == pytest.approx(134.41)
    assert lots[1] == pytest.approx(134.41)
    assert sum(lots) == pytest.approx(268.82)


def test_register_contract_lot_group_links_contract_ids():
    orch = type("Orch", (), {"_active_cycle_id": 57})()
    group_id = register_contract_lot_group(orch, [1001, 1002])
    assert group_id == 57
    assert orch._contract_lot_groups[57] == (1001, 1002)
    assert orch._contract_lot_group[1001] == 57


def test_resolve_fractional_lot_stagger_seconds_scales_with_rtt(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch.ws.last_rtt_seconds = 0.16
        delay = resolve_fractional_lot_stagger_seconds(orch)
        assert delay > 0.0


@pytest.mark.asyncio
async def test_dispatch_fractional_orders_staggers_split_proposals(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 57
        orch.ws.last_rtt_seconds = 0.10
        orch.ws.send = AsyncMock(
            side_effect=[
                {"proposal": {"id": "p-1", "ask_price": 134.41, "date_expiry": 1710000123, "payout": 260.0}},
                {"proposal": {"id": "p-2", "ask_price": 134.41, "date_expiry": 1710000124, "payout": 260.0}},
                {"buy": {"contract_id": 1001, "buy_price": 134.41, "payout": 260.0}},
                {"buy": {"contract_id": 1002, "buy_price": 134.41, "payout": 260.0}},
            ]
        )
        with (
            patch(
                "src.application.services.orchestrator.execution_fractional_lots_buy.subscribe_open_contract",
                new_callable=AsyncMock,
            ),
            patch(
                "src.application.services.orchestrator.execution_fractional_lots._stagger_fractional_dispatch",
                new_callable=AsyncMock,
            ) as stagger_mock,
        ):
            contracts = await dispatch_fractional_orders(
                orch.executor,
                "R_10",
                TradeDirection.CALL,
                268.82,
                duration=60,
                metrics={"duration": 60},
                order_n=1,
            )
        assert len(contracts) == 2
        stagger_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_stagger_fractional_dispatch_noop_for_first_lot(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")

        await _stagger_fractional_dispatch(orch, 0)
        assert resolve_fractional_lot_stagger_seconds(orch) >= 0.0


@pytest.mark.asyncio
async def test_dispatch_fractional_orders_single_lot_success(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        contract = MagicMock(contract_id=9001)
        orch.executor._place_order = AsyncMock(return_value=contract)
        with patch(
            "src.application.services.orchestrator.execution_fractional_lots.adopt_executed_contract",
            new_callable=AsyncMock,
        ) as adopt_mock:
            contracts = await dispatch_fractional_orders(
                orch.executor,
                "R_10",
                TradeDirection.CALL,
                150.0,
                duration=60,
                metrics={"duration": 60},
                order_n=1,
            )
        assert contracts == [contract]
        adopt_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_dispatch_fractional_orders_renews_proposal_per_split_lot(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 57
        orch.ws.send = AsyncMock(
            side_effect=[
                {"proposal": {"id": "p-1", "ask_price": 134.41, "date_expiry": 1710000123, "payout": 260.0}},
                {"proposal": {"id": "p-2", "ask_price": 134.41, "date_expiry": 1710000124, "payout": 260.0}},
                {"buy": {"contract_id": 1001, "buy_price": 134.41, "payout": 260.0}},
                {"buy": {"contract_id": 1002, "buy_price": 134.41, "payout": 260.0}},
            ]
        )
        orch.executor._place_order = AsyncMock()
        with patch(
            "src.application.services.orchestrator.execution_fractional_lots_buy.subscribe_open_contract",
            new_callable=AsyncMock,
        ) as subscribe_mock:
            contracts = await dispatch_fractional_orders(
                orch.executor,
                "R_10",
                TradeDirection.CALL,
                268.82,
                duration=60,
                metrics={"duration": 60},
                order_n=1,
            )
        assert len(contracts) == 2
        assert orch._contract_lot_groups[57] == (1001, 1002)
        orch.executor._place_order.assert_not_awaited()
        assert orch.ws.send.await_count == 4
        proposal_calls = [call.args[0] for call in orch.ws.send.await_args_list[:2]]
        assert proposal_calls[0]["proposal"] == 1
        assert proposal_calls[1]["proposal"] == 1
        assert proposal_calls[0]["passthrough"]["split_lot_id"] != proposal_calls[1]["passthrough"]["split_lot_id"]
        assert proposal_calls[0]["passthrough"]["split_batch_id"] == proposal_calls[1]["passthrough"]["split_batch_id"]
        assert proposal_calls[0]["passthrough"]["split_attempt_seq"] == 1
        assert proposal_calls[1]["passthrough"]["split_attempt_seq"] == 1
        assert proposal_calls[0]["passthrough"]["split_lot_index"] == 0
        assert proposal_calls[1]["passthrough"]["split_lot_index"] == 1
        assert orch._split_attempt_seq == 1
        assert orch._last_split_attempt_seq == 1
        buy_calls = [call.args[0] for call in orch.ws.send.await_args_list[2:]]
        assert buy_calls[0]["buy"] == "p-1"
        assert buy_calls[1]["buy"] == "p-2"
        assert subscribe_mock.call_count == 2
