"""Testes finais de cobertura para Orchestrator, run_loop, state_restore e settlement_queue_ops."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator import (
    Orchestrator,
    orchestrator_run_loop as orl,
    orchestrator_state_restore as osr,
    settlement_queue_ops as sqo,
)
from src.domain.models.trade import TradeDirection


@pytest.mark.asyncio
async def test_orchestrator_wrapper_methods_and_execution_delegates(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as ws_cls:
        ws_cls.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, auth=MagicMock())

    with (
        patch("src.application.services.orchestrator.bar_epoch_already_processed", AsyncMock(return_value=False)),
        patch("src.application.services.orchestrator.save_full_state", AsyncMock()),
        patch("src.application.services.orchestrator.persist_full_state_unlocked", AsyncMock()),
        patch("src.application.services.orchestrator.enqueue_contract_settlement"),
        patch("src.application.services.orchestrator.reconcile_single_contract", AsyncMock(return_value=True)),
        patch("src.application.services.orchestrator.post_settlement_cycle_pending", return_value=False),
        patch("src.application.services.orchestrator.run_post_settlement_breath_and_cycle", AsyncMock()),
        patch("src.application.services.orchestrator.schedule_trading_cycle_after_settlement"),
        patch("src.application.services.orchestrator.trading_cycle_entry_allowed", return_value=True),
        patch("src.application.services.orchestrator.run_trading_cycle_if_ready", AsyncMock(return_value=True)),
        patch("src.application.services.orchestrator.setup_session", AsyncMock(return_value=True)),
        patch("src.application.services.orchestrator.start_streams", AsyncMock(return_value=True)),
        patch("src.application.services.orchestrator.subscribe_transactions", AsyncMock()),
    ):
        assert await orch._bar_epoch_already_processed(300) is False
        await orch.save_full_state()
        await orch._persist_full_state_unlocked()
        orch._enqueue_contract_settlement(1, "R_10", 600)
        assert await orch._reconcile_single_contract(1) is True
        orch.mark_cluster_cycle_complete()
        assert orch._post_settlement_cycle_pending() is False
        await orch._run_post_settlement_breath_and_cycle()
        orch._schedule_trading_cycle_after_settlement()
        assert orch._trading_cycle_entry_allowed() is True
        assert await orch._run_trading_cycle_if_ready() is True
        orch._last_cluster_cycle_end = 1.0
        orch._last_idle_watchdog_attempt = 0.0
        await orch._tick_idle_cycle_watchdog()
        orch._last_interval_cycle_check = 0.0
        await orch._tick_interval_cycle_if_due()
        assert await orch.setup_session() is True
        assert await orch.start_streams() is True
        await orch.subscribe_transactions()
        orch.running = True
        orch.is_trading = False
        await orch._on_candle(None)
        with patch("src.application.services.orchestrator.enqueue_contract_settlement", AsyncMock()):
            await orch._on_contract_update({"proposal_open_contract": {"contract_id": 1}})
        orch.risk_manager.active_contract_ids = [77]
        await orch._on_transaction({"transaction": {"contract_id": 77}})
        await orch._on_transaction({"transaction": {"contract_id": 999}})
        orch.schedule_trading_cycle_after_settlement()
        await orch._save_full_state()

    from src.application.services.orchestrator.ws_bootstrap import subscribe_account_transactions

    orch_sub = SimpleNamespace(
        ws=SimpleNamespace(send=AsyncMock(side_effect=RuntimeError("sub fail")), subscribe=MagicMock()),
        _on_transaction=AsyncMock(),
        logger=MagicMock(),
    )
    await subscribe_account_transactions(orch_sub)

    # ExecutionManager delegates
    with (
        patch("src.application.services.orchestrator.execution_manager.place_order", AsyncMock(return_value=None)),
        patch("src.application.services.orchestrator.execution_manager.run_settlement_watch", AsyncMock()),
        patch(
            "src.application.services.orchestrator.execution_manager.wait_for_settlement", AsyncMock(return_value=True)
        ),
        patch("src.application.services.orchestrator.execution_manager.reconcile_contracts", AsyncMock()),
        patch("src.application.services.orchestrator.execution_manager.graceful_shutdown", AsyncMock()),
    ):
        assert await orch.executor._place_order("R_10", TradeDirection.CALL, 1.0) is None
        await orch.executor._run_settlement_watch()
        assert await orch.executor.wait_for_settlement(1.0) is True
        await orch.executor.reconcile_contracts()
        await orch.executor.shutdown(fast_path=True)


@pytest.mark.asyncio
async def test_orchestrator_run_loop_and_state_restore():
    orch = SimpleNamespace(
        config={"orchestrator": {}},
        metrics_server=SimpleNamespace(start=AsyncMock()),
        logger=MagicMock(),
        running=True,
        ws=SimpleNamespace(is_running=True),
        last_data_signature="sig",
        _cooldown_until=0.0,
        _post_settlement_incomplete_streak=0,
        _post_settlement_deadlock=False,
        _run_trading_cycle_if_ready=AsyncMock(),
        _tick_idle_cycle_watchdog=AsyncMock(),
        _tick_interval_cycle_if_due=AsyncMock(),
    )

    async def _stop_on_sleep(_sec):
        orch.running = False

    with (
        patch(
            "src.application.services.orchestrator.orchestrator_run_loop.setup_trading_session",
            AsyncMock(return_value=True),
        ),
        patch(
            "src.application.services.orchestrator.orchestrator_run_loop.start_orchestrator_streams",
            AsyncMock(return_value=True),
        ),
        patch(
            "src.application.services.orchestrator.orchestrator_run_loop.subscribe_account_transactions", AsyncMock()
        ),
        patch("src.application.services.orchestrator.orchestrator_run_loop.prepare_orchestrator_run_loop"),
        patch("src.application.services.orchestrator.orchestrator_run_loop.await_stream_warm_up_gate", AsyncMock()),
        patch("src.application.services.orchestrator.orchestrator_run_loop.start_settlement_worker", AsyncMock()),
        patch("src.application.services.orchestrator.orchestrator_run_loop.start_ingestion_watchdog", AsyncMock()),
        patch(
            "src.application.services.orchestrator.orchestrator_run_loop.get_data_state_signature", return_value="other"
        ),
        patch("src.application.services.orchestrator.orchestrator_run_loop.asyncio.sleep", side_effect=_stop_on_sleep),
    ):
        assert await orl.setup_session(orch) is True
        assert await orl.start_streams(orch) is True
        await orl.subscribe_transactions(orch)
        await orl.run_orchestrator_main_loop(orch)

    # state_restore
    store = SimpleNamespace(
        load_snapshot=AsyncMock(return_value={"risk": {"initial_bankroll": 50.0}, "total_session_profit": 5.0}),
        get_hash=AsyncMock(return_value={"R_10": "2.0"}),
        set_hash=AsyncMock(),
        get_string=AsyncMock(return_value="300"),
        set_string=AsyncMock(),
    )
    orch_s = SimpleNamespace(
        state_store=store,
        infra=SimpleNamespace(enabled=True),
        risk_manager=SimpleNamespace(restore_state=MagicMock(), total_session_profit=0.0),
        state_mgr=SimpleNamespace(
            state=SimpleNamespace(
                initial_balance=100.0,
                current_balance=105.0,
                daily_stop_win_target=0.0,
                total_trades_today=1,
                stop_win_triggered=False,
            )
        ),
        _session_targets_bootstrapped=False,
    )
    with (
        patch("src.application.services.orchestrator.orchestrator_state_restore.restore_session_hash", AsyncMock()),
        patch(
            "src.application.services.orchestrator.orchestrator_state_restore.restore_current_session_targets",
            AsyncMock(),
        ),
        patch(
            "src.application.services.orchestrator.orchestrator_state_restore.restore_market_signatures", AsyncMock()
        ),
    ):
        await osr.restore_orchestrator_state(orch_s)
        assert await osr.bar_epoch_already_processed(orch_s, "R_10", 300) is True
        store.get_string = AsyncMock(return_value=None)
        assert await osr.bar_epoch_already_processed(orch_s, "R_10", 300) is False
        await osr.persist_session_hash(orch_s)
        await osr.mark_bar_processed(orch_s, "R_10", 300)
        await osr.sync_market_signature(orch_s, "sig")
        orch_s.state_store = None
        await osr.restore_orchestrator_state(orch_s)
        await osr.persist_session_hash(orch_s)
        assert await osr.bar_epoch_already_processed(orch_s, "R_10", 300) is False
        await osr.mark_bar_processed(orch_s, "R_10", 300)
        await osr.sync_market_signature(orch_s, "")


@pytest.mark.asyncio
async def test_settlement_queue_ops_confirm_and_process():
    client = MagicMock(
        zrem=AsyncMock(),
        zrange=AsyncMock(
            return_value=[
                json.dumps({"proposal_open_contract": {}}),
                json.dumps({"proposal_open_contract": {"contract_id": 999}}),
                json.dumps({"proposal_open_contract": {"contract_id": 101}}),
                json.dumps({"proposal_open_contract": {"contract_id": 102}}),
            ]
        ),
    )
    orch = SimpleNamespace(
        state_store=SimpleNamespace(_redis=AsyncMock(return_value=client)),
        ws=SimpleNamespace(is_running=True),
        state=SimpleNamespace(active_contracts={101: object()}),
        risk_manager=SimpleNamespace(active_contract_ids=[101, 102], contract_to_symbol={}),
        logger=MagicMock(),
    )
    assert await sqo.get_redis_client(orch) is client

    with (
        patch(
            "src.application.services.orchestrator.settlement_backfill.fetch_open_contract",
            AsyncMock(side_effect=[{"contract_id": 101, "status": "WON", "is_settled": 1}, None]),
        ),
        patch(
            "src.application.services.orchestrator.settlement_backfill.settlement_payload_from_profit_row",
            return_value={"proposal_open_contract": {"contract_id": 102, "status": "WON", "is_settled": 1}},
        ),
        patch("src.application.services.orchestrator.settlement_logic.process_contract_settlement", AsyncMock()),
        patch(
            "src.application.services.orchestrator.settlement_logic.process_late_settlement_from_payload", AsyncMock()
        ),
        patch(
            "src.application.services.orchestrator.settlement_queue_ops.fetch_profit_table",
            AsyncMock(return_value=[{"contract_id": 102}]),
        ),
    ):
        await sqo.process_redis_settlement_queue(orch, force=True)
    assert client.zrem.await_count == 3

    # Erro na leitura Redis + rate-limit poll + offline
    client.zrange = AsyncMock(side_effect=RuntimeError("read err"))
    await sqo.process_redis_settlement_queue(orch, force=True)
    await sqo.process_redis_settlement_queue(orch, force=False)
    orch.ws.is_running = False
    await sqo.process_redis_settlement_queue(orch, force=True)
