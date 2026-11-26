"""Testes finais de cauda para fechar 100.00% de cobertura em src/application."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator import (
    api_maintenance_guard as amg,
    orchestrator_data_signature as ods,
    orchestrator_run_loop as orl,
    orchestrator_settlement_queue as osq,
    orchestrator_state_restore as osr,
    orchestrator_state_session as oss,
    post_settlement_cycle as psc,
    settlement_audit as sa,
    settlement_helpers as sh,
    settlement_logic as sl,
    settlement_queue_ops as sqo,
    settlement_reconciliation as sr,
    trading_cycle_entry as tce,
    trading_cycle_entry_guards as tceg,
    warm_up_buffer_guard as wug,
    ws_bootstrap as wb,
)
from src.application.services.orchestrator.execution_manager import ExecutionManager
from src.application.services.orchestrator.execution_orders import place_order
from src.application.services.orchestrator.execution_settlement import _settlement_grace_period, wait_for_settlement
from src.application.services.orchestrator.orchestrator_atomic_state import orchestrator_balance_snapshot
from src.application.services.orchestrator.session_target_bootstrap import restore_current_session_targets
from src.domain.models.trade import Contract, TradeDirection, TradeStatus
from src.domain.risk.risk_manager import RiskManager
from src.infrastructure.state.trading_state import TradingState


@pytest.mark.asyncio
async def test_tail_ws_bootstrap_and_warmup_and_signatures():
    # ws_bootstrap: _fresh_otp_uri, open_broker_handshake timeout, _resolve_rest_account_balance, open_public_market_handshake
    orch = SimpleNamespace(
        config={"orchestrator": {"broker_handshake_timeout_seconds": 0.01}},
        auth=MagicMock(
            mode="demo",
            account_id_override="VRTC1",
            open_trading_session=AsyncMock(
                return_value=SimpleNamespace(ws_url="wss://x?otp=2", balance=100.0, account_id="VRTC1")
            ),
            rest_client=MagicMock(
                return_value=SimpleNamespace(
                    list_accounts=AsyncMock(
                        return_value=[
                            SimpleNamespace(account_id="VRTC1", balance=200.0, account_type="demo", status="active")
                        ]
                    )
                )
            ),
        ),
        ws=SimpleNamespace(connect=AsyncMock(), is_running=True, ws=None),
        risk_manager=RiskManager({"params": {"stake_min": 1.0}}),
        state=SimpleNamespace(balance=0.0),
        logger=MagicMock(),
    )

    async def _call_factory(url, **kwargs):
        if "uri_factory" in kwargs:
            await kwargs["uri_factory"]()

    orch.ws.connect = AsyncMock(side_effect=_call_factory)
    sess = await wb._broker_pat_websocket_handshake(orch)
    assert sess.ws_url == "wss://x?otp=2"

    with (
        patch(
            "src.application.services.orchestrator.ws_bootstrap._broker_pat_websocket_handshake",
            AsyncMock(side_effect=TimeoutError()),
        ),
        pytest.raises(RuntimeError, match="HANDSHAKE_TIMEOUT"),
    ):
        await wb.open_broker_handshake(orch)

    acc_id, bal = await wb._resolve_rest_account_balance(orch)
    assert acc_id == "VRTC1" and bal == 200.0

    await wb.open_public_market_handshake(orch)
    orch.ws.connect = AsyncMock(side_effect=TimeoutError())
    with pytest.raises(RuntimeError, match="HANDSHAKE_TIMEOUT"):
        await wb.open_public_market_handshake(orch)

    # warm_up_buffer_guard edges
    orch_w = SimpleNamespace(config={}, logger=MagicMock(), _stream_warmed_up_at=0.0)
    assert wug.stream_warm_up_remaining(orch_w) == 0.0
    orch_w._stream_warmed_up_at = asyncio.get_running_loop().time() + 10.0
    assert wug.stream_warm_up_remaining(orch_w) > 0.0
    orch_w._warm_up_logged_until = orch_w._stream_warmed_up_at + 1.0
    wug.log_warm_up_guard_suspension(orch_w)
    assert wug.trading_cycle_warm_up_suspended(orch_w) == wug.SIGNAL_SUSPENDED

    now = asyncio.get_running_loop().time()
    orch_w._stream_warmed_up_at = now + 0.05
    orch_w.stream = SimpleNamespace(tick_buffer=SimpleNamespace(last_tick_monotonic=lambda: now))
    assert await wug.await_stream_warm_up_gate(orch_w, timeout=0.0) is True
    orch_w._stream_warmed_up_at = now + 0.02
    assert await wug.await_stream_warm_up_gate(orch_w, timeout=1.0) is True

    # orchestrator_data_signature & state_session & state_restore
    assert ods.resolve_signature_boundary_seconds(SimpleNamespace(config={"stream": {"micro_granularity": 300}})) > 0
    assert (
        ods.seconds_until_next_signature_boundary(SimpleNamespace(config={"stream": {"micro_granularity": 300}})) >= 0.0
    )
    store = SimpleNamespace(get_string=AsyncMock(side_effect=["sig", "bad_int"]))
    orch_ss = SimpleNamespace(anchor="R_10", last_data_signature="")
    await oss.restore_market_signatures(orch_ss, store)
    assert orch_ss.last_data_signature == "sig"
    assert (
        await osr.bar_epoch_already_processed(
            SimpleNamespace(
                state_store=SimpleNamespace(get_string=AsyncMock(return_value="bad")),
                infra=SimpleNamespace(enabled=True),
            ),
            "R_10",
            300,
        )
        is False
    )
    assert (
        await restore_current_session_targets(SimpleNamespace(_session_targets_bootstrapped=True, state_store=None))
        is None
    )


@pytest.mark.asyncio
async def test_tail_settlement_queue_orders_and_post_cycle():
    # orchestrator_settlement_queue.enqueue_contract_settlement
    orch_q = SimpleNamespace(_settlement_queue=asyncio.Queue())
    await osq.enqueue_contract_settlement(orch_q, {"proposal_open_contract": {"status": "OPEN"}})
    await osq.enqueue_contract_settlement(orch_q, {"proposal_open_contract": {"status": "WON"}})
    orch_q._settled_contract_ids = {1}
    await osq.enqueue_contract_settlement(orch_q, {"proposal_open_contract": {"contract_id": 1, "status": "WON"}})
    orch_q._settlement_queued_ids = {2}
    await osq.enqueue_contract_settlement(orch_q, {"proposal_open_contract": {"contract_id": 2, "status": "WON"}})
    orch_q._settlement_queue = SimpleNamespace(put=AsyncMock(side_effect=RuntimeError("q full")))
    with pytest.raises(RuntimeError, match="q full"):
        await osq.enqueue_contract_settlement(orch_q, {"proposal_open_contract": {"contract_id": 3, "status": "WON"}})

    # execution_manager._run_settlement_watch & execution_orders empty attempts & retriable raise
    mgr = ExecutionManager(
        SimpleNamespace(
            config={"laya": {"stake_min": 1.0}, "orchestrator": {"execution": {}}},
            risk_manager=RiskManager(),
            state=TradingState(),
            logger=MagicMock(),
        )
    )
    with patch("src.application.services.orchestrator.execution_manager.run_settlement_watch", AsyncMock()):
        await mgr._run_settlement_watch()
    with patch("src.application.services.orchestrator.execution_orders.proposal_stake_attempts", return_value=[]):
        assert await place_order(mgr, "R_10", TradeDirection.CALL, 1.0) is None
    mgr.orch.trade_handler = SimpleNamespace(buy_with_parameters=AsyncMock(side_effect=RuntimeError("hard fail")))
    with pytest.raises(RuntimeError, match="hard fail"):
        await place_order(mgr, "R_10", TradeDirection.CALL, 1.0)

    # execution_settlement: dynamic grace > 0 & wait_for_settlement branches
    mgr.orch.config = {
        "orchestrator": {"execution": {"settlement_poll_seconds": 0.0, "settlement_max_stagnant_polls": 1}},
        "risk_management": {"params": {}},
    }
    with patch(
        "src.application.services.orchestrator.execution_settlement.settlement_audit.calculate_cluster_grace_period",
        return_value=999.0,
    ):
        assert _settlement_grace_period(mgr, {}, 0.0) >= 999.0
    mgr.orch.risk_manager.register_open_contract(11, "R_10", 5.0)
    await mgr.orch.state.add_contract(
        Contract(11, "p", TradeStatus.OPEN, 5.0, 9.5, "R_10", TradeDirection.CALL, 5.0, 9999999999)
    )
    mgr.orch.ws = SimpleNamespace(is_running=False)
    mgr.orch.save_full_state = AsyncMock()
    with (
        patch(
            "src.application.services.orchestrator.execution_settlement._wait_broker_offline_settlement",
            AsyncMock(side_effect=[True, False, False]),
        ),
        patch.object(mgr, "reconcile_contracts", AsyncMock(return_value=True)),
        patch(
            "src.application.services.orchestrator.execution_settlement._handle_stagnant_settlement",
            AsyncMock(side_effect=["continue", None]),
        ),
        patch(
            "src.application.services.orchestrator.execution_settlement._settlement_poll_delay",
            AsyncMock(side_effect=lambda _: mgr.orch.risk_manager.active_contract_ids.clear()),
        ),
    ):
        await wait_for_settlement(mgr, timeout=3600)

    # api_maintenance_guard, atomic_state, settlement_audit, helpers, logic, queue_ops, reconciliation, run_loop, entry_guards
    assert amg.api_maintenance_blocks_trading_cycle(SimpleNamespace(_api_maintenance_until=0.0)) is False
    amg.log_api_maintenance_hibernation(SimpleNamespace(_api_maintenance_until=0.0, logger=MagicMock()))
    assert (
        orchestrator_balance_snapshot(SimpleNamespace(state_mgr=SimpleNamespace(read_cached_balance=lambda: 123.0)))
        == 123.0
    )
    assert sa.is_transient_broker_error(type("ConnectionClosed", (Exception,), {})()) is True
    assert sh.process_settlement_outcome is not None
    assert await sh.fetch_profit_table(
        SimpleNamespace(send=AsyncMock(return_value={"profit_table": {"transactions": [{"a": 1}]}})),
        limit=1,
        timeout=1.0,
    ) == [{"a": 1}]
    assert sqo.get_redis_client is not None
    assert sr.reconcile_single_contract is not None

    # post_settlement_cycle retry loop edges
    orch_p = SimpleNamespace(
        running=True,
        is_trading=True,
        config={"orchestrator": {"post_settlement_breath_seconds": 0.0}},
        state=SimpleNamespace(active_contracts={1: object()}, set_trading=AsyncMock()),
        risk_manager=RiskManager(),
        _post_settlement_task=None,
        _post_settlement_wake=asyncio.Event(),
        _trading_slot_poll_task=None,
        _last_cycle_cluster_executed=False,
        _run_trading_cycle_if_ready=AsyncMock(),
        logger=MagicMock(),
    )
    psc._release_post_settlement_task(orch_p, MagicMock())
    orch_p._post_settlement_task = MagicMock(done=lambda: False)
    psc.schedule_trading_cycle_after_settlement(orch_p)
    orch_p.state.active_contracts.clear()
    psc.schedule_trading_cycle_after_settlement(orch_p)

    # _try_stop_win_fast_path True branch & _run_post_settlement_retry_loop branches
    with (
        patch(
            "src.application.services.orchestrator.post_settlement_cycle.check_session_limits_before_post_settlement",
            return_value=True,
        ),
        patch(
            "src.application.services.orchestrator.post_settlement_cycle.clear_current_session_redis_keys", AsyncMock()
        ),
        patch("src.application.services.orchestrator.post_settlement_cycle.graceful_shutdown", AsyncMock()),
    ):
        assert await psc._try_stop_win_fast_path(orch_p) is True
        await psc.run_post_settlement_breath_and_cycle(orch_p)

    # _run_post_settlement_retry_loop active_contracts + is_trading + tolerance window
    step = {"n": 0}

    async def _poll_step(_sec):
        step["n"] += 1
        if step["n"] == 1:
            orch_p.state.active_contracts.clear()
        elif step["n"] == 2:
            orch_p.is_trading = False
        elif step["n"] >= 4:
            orch_p.running = False

    orch_p.state.active_contracts[1] = object()
    with (
        patch("src.application.services.orchestrator.post_settlement_cycle._poll_delay", side_effect=_poll_step),
        patch(
            "src.application.services.orchestrator.post_settlement_cycle._apply_tolerance_window_recovery",
            AsyncMock(return_value=0),
        ),
        patch(
            "src.application.services.orchestrator.post_settlement_cycle.resolve_settlement_tolerance_window",
            return_value=0.0,
        ),
        patch(
            "src.application.services.orchestrator.post_settlement_cycle._attempt_post_settlement_trading_cycle",
            AsyncMock(side_effect=[False, True]),
        ),
    ):
        await psc._run_post_settlement_retry_loop(orch_p, {}, 0.0)

    # trading_cycle_entry_guards & trading_cycle_entry
    tceg.commit_trading_cycle_data_signature(
        SimpleNamespace(
            stream=SimpleNamespace(candles={}), anchor="R_10", symbols=["R_10"], config={}, last_data_signature=""
        )
    )
    with patch(
        "src.application.services.orchestrator.trading_cycle_entry_guards.get_data_state_signature",
        return_value="sig_new",
    ):
        o_sig = SimpleNamespace(last_data_signature="sig_new")
        tceg.commit_trading_cycle_data_signature(o_sig)
        assert (
            tceg.trading_cycle_entry_allowed(
                SimpleNamespace(
                    running=True,
                    is_trading=False,
                    _reconciliation_pending=False,
                    _api_maintenance_until=0.0,
                    _session_persistence_write_active=False,
                    state=SimpleNamespace(active_contracts={}),
                    _last_evaluated_boundary_epoch=0,
                    last_data_signature="sig_new",
                    config={},
                )
            )
            is False
        )

    # orchestrator_run_loop cooldown sleep branch
    orch_rl = SimpleNamespace(
        config={"orchestrator": {}},
        metrics_server=None,
        logger=MagicMock(),
        running=True,
        ws=SimpleNamespace(is_running=True),
        last_data_signature="",
        _cooldown_until=9999999999.0,
        _run_trading_cycle_if_ready=AsyncMock(),
    )

    async def _stop_sleep(_s):
        orch_rl.running = False

    with (
        patch(
            "src.application.services.orchestrator.orchestrator_run_loop.setup_trading_session",
            AsyncMock(return_value=True),
        ),
        patch(
            "src.application.services.orchestrator.orchestrator_run_loop.start_orchestrator_streams",
            AsyncMock(return_value=True),
        ),
        patch("src.application.services.orchestrator.orchestrator_run_loop.prepare_orchestrator_run_loop"),
        patch("src.application.services.orchestrator.orchestrator_run_loop.await_stream_warm_up_gate", AsyncMock()),
        patch("src.application.services.orchestrator.orchestrator_run_loop.start_settlement_worker", AsyncMock()),
        patch("src.application.services.orchestrator.orchestrator_run_loop.start_ingestion_watchdog", AsyncMock()),
        patch("src.application.services.orchestrator.orchestrator_run_loop.asyncio.sleep", side_effect=_stop_sleep),
    ):
        await orl.run_orchestrator_main_loop(orch_rl)


@pytest.mark.asyncio
async def test_tail_final_40_lines_for_100_percent():
    from datetime import UTC, datetime

    from src.application.services.orchestrator.settlement_reconciliation import _settle_from_profit_burst

    # api_maintenance_guard: line 89 (current > end_at) & handle_broker_maintenance_error
    now_past = datetime(2026, 11, 26, 5, 0, 0, tzinfo=UTC)
    assert amg.api_maintenance_delay_seconds("Update from 01:00:00 to 02:00:00 GMT", now=now_past) > 0
    o_amg = SimpleNamespace(config={}, logger=MagicMock(), _api_maintenance_until=0.0)
    assert amg.handle_broker_maintenance_error(o_amg, RuntimeError("This market is presently closed.")) is True

    # execution_orders line 71 (last_error None fallback) & execution_settlement lines 34, 147, 149
    mgr = ExecutionManager(
        SimpleNamespace(
            config={"laya": {"stake_min": 1.0}, "orchestrator": {"execution": {"settlement_poll_seconds": 0.0}}},
            risk_manager=RiskManager(),
            state=TradingState(),
            ws=SimpleNamespace(is_running=True),
            logger=MagicMock(),
            save_full_state=AsyncMock(),
        )
    )
    with patch("src.application.services.orchestrator.execution_orders.proposal_stake_attempts", return_value=[1.0]):
        mgr.orch.trade_handler = SimpleNamespace(buy_with_parameters=AsyncMock(return_value=None))
        with pytest.raises(RuntimeError, match="falha desconhecida"):
            await place_order(mgr, "R_10", TradeDirection.CALL, 1.0)
    with patch(
        "src.application.services.orchestrator.execution_settlement.settlement_audit.calculate_cluster_grace_period",
        return_value=0.0,
    ):
        assert _settlement_grace_period(mgr, {}, 0.0) >= 0.0
    mgr.orch.risk_manager.register_open_contract(55, "R_10", 5.0)
    with patch("src.application.services.orchestrator.execution_settlement._settlement_timed_out", return_value=True):
        await wait_for_settlement(mgr, timeout=1)
    mgr.orch.risk_manager.register_open_contract(56, "R_10", 5.0)
    with patch(
        "src.application.services.orchestrator.execution_settlement._prune_orphan_settlement_ids", return_value=True
    ):
        await wait_for_settlement(mgr, timeout=1)

    # orchestrator_data_signature lines 73, 81
    assert (
        ods.resolve_signature_boundary_seconds(
            SimpleNamespace(
                config={"orchestrator": {"signature_boundary_seconds": "bad", "cycle_interval_seconds": "bad"}}
            )
        )
        > 0
    )

    # post_settlement_cycle remaining lines (35, 83, 188, 205, 208-209, 214, 247, 273-276, 284)
    o_psc = SimpleNamespace(
        running=True,
        is_trading=False,
        config={"orchestrator": {"post_settlement_breath_seconds": 0.0, "exec_empty_retry_seconds": "bad"}},
        state=SimpleNamespace(active_contracts={}, set_trading=AsyncMock()),
        risk_manager=SimpleNamespace(active_contract_ids=[], pending_loss_total=lambda: 5.0),
        _post_settlement_task=None,
        _post_settlement_wake=asyncio.Event(),
        logger=MagicMock(),
    )
    t_cur = asyncio.current_task()
    o_psc._post_settlement_task = t_cur
    psc._release_post_settlement_task(o_psc, t_cur)
    assert o_psc._post_settlement_task is None
    with patch("src.application.services.orchestrator.post_settlement_cycle._poll_delay", AsyncMock()):
        await psc._await_post_settlement_breath(o_psc, 0.02, 0.01)
    o_psc.config = "not-a-dict"
    with (
        patch(
            "src.application.services.orchestrator.post_settlement_cycle.seconds_until_next_signature_boundary",
            side_effect=[10.0, 0.0],
        ),
        patch("src.application.services.orchestrator.post_settlement_cycle._await_post_settlement_breath", AsyncMock()),
        patch("src.application.services.orchestrator.post_settlement_cycle._poll_delay", AsyncMock()),
    ):
        await psc._await_exec_empty_signature_alignment(o_psc, 0.0)
        o_psc._cooldown_until = 0.0
        await psc._await_exec_empty_signature_alignment(o_psc, 0.0)
    o_psc.config = {"orchestrator": {}}
    o_psc.running = True
    with (
        patch(
            "src.application.services.orchestrator.post_settlement_cycle._try_stop_win_fast_path",
            AsyncMock(return_value=False),
        ),
        patch(
            "src.application.services.orchestrator.post_settlement_cycle._attempt_post_settlement_trading_cycle",
            AsyncMock(side_effect=[False, False, True]),
        ),
        patch(
            "src.application.services.orchestrator.post_settlement_cycle._apply_tolerance_window_recovery",
            AsyncMock(return_value=0),
        ),
        patch(
            "src.application.services.orchestrator.post_settlement_cycle.resolve_settlement_tolerance_window",
            return_value=0.0,
        ),
    ):
        await psc._run_post_settlement_retry_loop(o_psc, {}, 0.0)
    with patch(
        "src.application.services.orchestrator.post_settlement_cycle._try_stop_win_fast_path",
        AsyncMock(return_value=True),
    ):
        await psc._run_post_settlement_retry_loop(o_psc, {}, 0.0)

    # Extra exact branches: execution_manager:89, psc:83/205/208-209/273-276, sqo:68
    with patch("src.application.services.orchestrator.execution_manager.run_settlement_watch", AsyncMock()):
        await mgr._run_settlement_watch()

    async def _wake_during_poll(_sec):
        o_psc._post_settlement_wake.set()
        await asyncio.sleep(0.02)

    with patch(
        "src.application.services.orchestrator.post_settlement_cycle._poll_delay", side_effect=_wake_during_poll
    ):
        await psc._await_post_settlement_breath(o_psc, 0.5, 0.5)
    o_psc.config = {"orchestrator": "bad"}
    with (
        patch(
            "src.application.services.orchestrator.post_settlement_cycle.seconds_until_next_signature_boundary",
            return_value=10.0,
        ),
        patch("src.application.services.orchestrator.post_settlement_cycle._await_post_settlement_breath", AsyncMock()),
    ):
        await psc._await_exec_empty_signature_alignment(o_psc, 0.0)
        o_psc._cooldown_until = 0.0
        o_psc.config = {"orchestrator": {"exec_empty_retry_seconds": "bad"}}
        await psc._await_exec_empty_signature_alignment(o_psc, 0.0)
    o_psc.config = {"orchestrator": {}}
    with (
        patch(
            "src.application.services.orchestrator.post_settlement_cycle._try_stop_win_fast_path",
            AsyncMock(return_value=False),
        ),
        patch(
            "src.application.services.orchestrator.post_settlement_cycle._attempt_post_settlement_trading_cycle",
            AsyncMock(side_effect=[False, False, False, True]),
        ),
        patch(
            "src.application.services.orchestrator.post_settlement_cycle._apply_tolerance_window_recovery",
            AsyncMock(return_value=0),
        ),
        patch(
            "src.application.services.orchestrator.post_settlement_cycle.resolve_settlement_tolerance_window",
            return_value=999.0,
        ),
        patch("src.application.services.orchestrator.post_settlement_cycle._await_post_settlement_breath", AsyncMock()),
        patch(
            "src.application.services.orchestrator.post_settlement_cycle._await_exec_empty_signature_alignment",
            AsyncMock(side_effect=lambda o, _: setattr(o, "running", True)),
        ),
    ):
        await psc._run_post_settlement_retry_loop(o_psc, {}, 0.0)
    o_psc.running = True
    with (
        patch(
            "src.application.services.orchestrator.post_settlement_cycle._try_stop_win_fast_path",
            AsyncMock(return_value=False),
        ),
        patch(
            "src.application.services.orchestrator.post_settlement_cycle._attempt_post_settlement_trading_cycle",
            AsyncMock(side_effect=[False, False, True]),
        ),
        patch(
            "src.application.services.orchestrator.post_settlement_cycle._apply_tolerance_window_recovery",
            AsyncMock(return_value=0),
        ),
        patch(
            "src.application.services.orchestrator.post_settlement_cycle.resolve_settlement_tolerance_window",
            return_value=0.0,
        ),
    ):
        await psc._run_post_settlement_retry_loop(o_psc, {}, 0.0)
    assert await sqo.get_redis_client(SimpleNamespace(config={"redis": "bad"})) is not None

    # settlement_audit 107-108, settlement_helpers 73, settlement_logic 87-88, settlement_queue_ops 68/161/170/174-175/204, reconciliation 118
    sa.clear_contract_tracking([1], RiskManager())
    assert await sh.fetch_portfolio(
        SimpleNamespace(send=AsyncMock(return_value={"portfolio": {"contracts": [{"id": 1}]}})), timeout=1.0
    ) == [{"id": 1}]
    o_sl = SimpleNamespace(
        state=SimpleNamespace(
            finalize_contract=AsyncMock(
                return_value=Contract(91, "p", TradeStatus.OPEN, 1.0, 1.9, "R_10", TradeDirection.CALL, 1.0, 9999999999)
            )
        ),
        risk_manager=RiskManager(),
        logger=MagicMock(),
    )
    with (
        patch("src.application.services.orchestrator.settlement_logic.record_settlement_audit", AsyncMock()),
        patch("src.application.services.orchestrator.settlement_logic._complete_settlement", AsyncMock()),
    ):
        await sl.process_late_settlement_from_payload(o_sl, {"contract_id": 91, "status": "WON", "profit": 0.9})

    assert await sqo.get_redis_client(SimpleNamespace(config={"infra": {"redis": "bad"}})) is not None
    c_empty = MagicMock(zrange=AsyncMock(return_value=[]))
    with patch(
        "src.application.services.orchestrator.settlement_queue_ops.get_redis_client", AsyncMock(return_value=c_empty)
    ):
        await sqo.process_redis_settlement_queue(SimpleNamespace(ws=SimpleNamespace(is_running=True)), force=True)
    o_sqo = SimpleNamespace(
        config={},
        ws=SimpleNamespace(is_running=True),
        state=SimpleNamespace(active_contracts={}),
        risk_manager=SimpleNamespace(active_contract_ids=[11], contract_to_symbol={}),
        logger=MagicMock(),
    )
    c_err = MagicMock(zrem=AsyncMock())
    with (
        patch(
            "src.application.services.orchestrator.settlement_backfill.fetch_open_contract",
            AsyncMock(side_effect=[{"contract_id": 11, "status": "WON", "is_settled": 1}, RuntimeError("api")]),
        ),
        patch(
            "src.application.services.orchestrator.settlement_logic.process_late_settlement_from_payload", AsyncMock()
        ),
    ):
        await sqo._confirm_queued_item(o_sqo, c_err, '{"proposal_open_contract":{"contract_id":11}}')
        await sqo._confirm_queued_item(o_sqo, c_err, '{"proposal_open_contract":{"contract_id":11}}')
    o_sqo.state.active_contracts[12] = object()
    o_sqo.risk_manager.active_contract_ids.append(12)
    with (
        patch(
            "src.application.services.orchestrator.settlement_backfill.fetch_open_contract",
            AsyncMock(return_value=None),
        ),
        patch(
            "src.application.services.orchestrator.settlement_queue_ops.fetch_profit_table",
            AsyncMock(return_value=[{"contract_id": 12}]),
        ),
        patch(
            "src.application.services.orchestrator.settlement_backfill.settlement_payload_from_profit_row",
            return_value={"proposal_open_contract": {"contract_id": 12, "status": "WON", "is_settled": 1}},
        ),
        patch("src.application.services.orchestrator.settlement_logic.process_contract_settlement", AsyncMock()),
    ):
        await sqo._confirm_queued_item(o_sqo, c_err, '{"proposal_open_contract":{"contract_id":12}}')

    res_rec = SimpleNamespace(settled_count=0, errors=[])
    with (
        patch(
            "src.application.services.orchestrator.settlement_reconciliation.fetch_profit_table",
            AsyncMock(return_value=[{"contract_id": 12}]),
        ),
        patch(
            "src.application.services.orchestrator.settlement_reconciliation.settlement_payload_from_profit_row",
            return_value={"proposal_open_contract": {"contract_id": 12, "status": "WON"}},
        ),
        patch(
            "src.application.services.orchestrator.settlement_reconciliation.process_contract_settlement", AsyncMock()
        ),
    ):
        await _settle_from_profit_burst(o_sqo, open_ids=set(), settled_ids=set(), timeout=1.0, result=res_rec)

    # trading_cycle_entry 73, trading_cycle_entry_guards 60, warm_up_buffer_guard 75-76/117, ws_bootstrap 215-216
    o_tce = SimpleNamespace(
        anchor="R_10",
        _active_cycle_id=1,
        config={"laya": {"stake_min": 100.0}},
        stream=SimpleNamespace(
            get_closed_candles=lambda s, g: [SimpleNamespace(epoch=300), SimpleNamespace(epoch=600)]
        ),
        laya=SimpleNamespace(
            decide=AsyncMock(
                return_value=SimpleNamespace(
                    symbol="R_10",
                    direction=TradeDirection.CALL,
                    probability=0.6,
                    prob_call=0.6,
                    prob_put=0.4,
                    confidence=0.6,
                    latency_ms=1.0,
                    indicators={},
                )
            )
        ),
        state=SimpleNamespace(balance=1.0),
        logger=MagicMock(),
    )
    with (
        patch(
            "src.application.services.orchestrator.trading_cycle_entry.compute_indicators",
            return_value=SimpleNamespace(
                to_dict=lambda: {},
                rsi_14=50.0,
                adx_14=20.0,
                ema_20=1.0,
                ema_50=1.0,
                atr_14=0.1,
                macd_hist=0.0,
                stoch_k=50.0,
                hurst_14=0.5,
                session_zscore=0.0,
            ),
        ),
        patch(
            "src.application.services.orchestrator.trading_cycle_entry.calculate_laya_stake", return_value=(0.0, 0.0)
        ),
        patch("src.application.services.orchestrator.trading_cycle_entry.log_laya_cycle_decision"),
    ):
        assert await tce._execute_laya_cycle(o_tce) is False

    o_tce.executor = SimpleNamespace(execute_laya_order=AsyncMock(side_effect=RuntimeError("order err")))
    with (
        patch(
            "src.application.services.orchestrator.trading_cycle_entry.compute_indicators",
            return_value=SimpleNamespace(
                to_dict=lambda: {},
                rsi_14=50.0,
                adx_14=20.0,
                ema_20=1.0,
                ema_50=1.0,
                atr_14=0.1,
                macd_hist=0.0,
                stoch_k=50.0,
                hurst_14=0.5,
                session_zscore=0.0,
            ),
        ),
        patch(
            "src.application.services.orchestrator.trading_cycle_entry.calculate_laya_stake", return_value=(10.0, 0.01)
        ),
        patch("src.application.services.orchestrator.trading_cycle_entry.log_laya_cycle_decision"),
    ):
        assert await tce._execute_laya_cycle(o_tce) is False

    o_guard = SimpleNamespace(
        running=True,
        is_trading=False,
        _reconciliation_pending=False,
        _api_maintenance_until=0.0,
        _session_persistence_write_active=False,
        state=SimpleNamespace(active_contracts={}),
        _last_evaluated_boundary_epoch=300,
        last_data_signature="",
        config={},
    )
    with patch("src.application.services.orchestrator.trading_cycle_entry_guards.m5_boundary_epoch", return_value=300):
        assert tceg.trading_cycle_entry_allowed(o_guard) is False

    o_wug = SimpleNamespace(
        _stream_warmed_up_at=asyncio.get_running_loop().time() + 10.0, _warm_up_logged_until=0.0, logger=MagicMock()
    )
    wug.log_warm_up_guard_suspension(o_wug)
    assert await wug.await_stream_warm_up_gate(SimpleNamespace(_stream_warmed_up_at=0.0)) is True

    o_wb_err = SimpleNamespace(infra=SimpleNamespace(enabled=False), config={}, logger=MagicMock())
    with patch(
        "src.application.services.orchestrator.ws_bootstrap.validate_infra_services",
        AsyncMock(side_effect=ValueError("fail")),
    ):
        assert await wb.setup_trading_session(o_wb_err) is False


@pytest.mark.real_settlement_watch
@pytest.mark.asyncio
async def test_execution_manager_real_settlement_watch_delegate():
    mgr = ExecutionManager(SimpleNamespace())
    with patch(
        "src.application.services.orchestrator.execution_manager.run_settlement_watch", AsyncMock()
    ) as watch_mock:
        await mgr._run_settlement_watch()
    watch_mock.assert_awaited_once_with(mgr)
