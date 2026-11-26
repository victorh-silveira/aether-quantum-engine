"""Testes unitarios para fechar 100% em settlement e post_settlement_cycle."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator import (
    post_settlement_cycle as psc,
    settlement_audit as sa,
    settlement_helpers as sh,
    settlement_logic as sl,
    settlement_queue_ops as sqo,
)
from src.domain.models.trade import Contract, TradeDirection, TradeStatus
from src.domain.risk.risk_manager import RiskManager
from src.infrastructure.state.state_manager import StateManager
from src.infrastructure.state.trading_state import TradingState


def _orch():
    TradingState.reset()
    state = TradingState()
    state.balance = 1000.0
    return SimpleNamespace(
        config={
            "trading": {"mode": "demo"},
            "orchestrator": {"post_settlement_breath_seconds": 0.0, "post_settlement_cycle_timeout_seconds": 1.0},
            "infra": {"redis": {"url": "redis://127.0.0.1:6379/0"}},
        },
        state=state,
        risk_manager=RiskManager({"params": {"stake_min": 1.0}}),
        state_mgr=StateManager(),
        ws=MagicMock(is_running=True),
        market_writer=SimpleNamespace(enqueue_contract_audit=AsyncMock()),
        logger=MagicMock(),
        running=True,
        is_trading=False,
        _active_cycle_id=1,
        _contract_cycle={101: 1},
        _cluster_results=[],
        _session_wins=0,
        _session_losses=0,
        _last_result_cycle_id=0,
        _post_settlement_task=None,
        _post_settlement_wake=asyncio.Event(),
        _last_cycle_cluster_executed=True,
        _persist_full_state_unlocked=AsyncMock(),
        _run_trading_cycle_if_ready=AsyncMock(),
        schedule_trading_cycle_after_settlement=MagicMock(),
    )


@pytest.mark.asyncio
async def test_settlement_logic_helpers_and_audit():
    orch = _orch()
    contract = Contract(101, "p1", TradeStatus.OPEN, 10.0, 19.5, "R_10", TradeDirection.CALL, 10.0, 9999999999)
    await orch.state.add_contract(contract)
    orch.risk_manager.register_open_contract(101, "R_10", 10.0)

    # record_settlement_audit success and error
    await sa.record_settlement_audit(orch, {"contract_id": 101}, contract, "R_10")
    orch.market_writer.enqueue_contract_audit.side_effect = RuntimeError("db")
    await sa.record_settlement_audit(orch, {"contract_id": 101}, contract, "R_10")
    await sa.record_settlement_audit(SimpleNamespace(), {}, contract, "R_10")

    # log_cluster_summary + fetch_portfolio/profit_table error paths
    sh.log_cluster_summary(orch)
    ws_bad = SimpleNamespace(send=AsyncMock(return_value={"error": {"message": "x"}}))
    assert await sh.fetch_portfolio(ws_bad, timeout=1.0) == []
    assert await sh.fetch_profit_table(ws_bad, limit=5, timeout=1.0) == []
    ws_bad2 = SimpleNamespace(send=AsyncMock(return_value={"portfolio": "bad", "profit_table": "bad"}))
    assert await sh.fetch_portfolio(ws_bad2, timeout=1.0) == []
    assert await sh.fetch_profit_table(ws_bad2, limit=5, timeout=1.0) == []
    ws_bad3 = SimpleNamespace(
        send=AsyncMock(return_value={"portfolio": {"contracts": "bad"}, "profit_table": {"transactions": "bad"}})
    )
    assert await sh.fetch_portfolio(ws_bad3, timeout=1.0) == []
    assert await sh.fetch_profit_table(ws_bad3, limit=5, timeout=1.0) == []

    # process_contract_settlement: missing key, open, missing cid, offline -> push_to_redis, happy path, duplicate
    await sl.process_contract_settlement(orch, {})
    await sl.process_contract_settlement(orch, {"proposal_open_contract": {"status": "OPEN"}})
    await sl.process_contract_settlement(orch, {"proposal_open_contract": {"status": "WON"}})
    orch.ws.is_running = False
    with patch(
        "src.application.services.orchestrator.settlement_logic.push_to_redis_priority_queue", AsyncMock()
    ) as push_m:
        await sl.process_contract_settlement(
            orch, {"proposal_open_contract": {"contract_id": 101, "status": "WON", "profit": 9.5}}
        )
        push_m.assert_awaited_once()
    orch.ws.is_running = True
    await sl.process_contract_settlement(
        orch, {"proposal_open_contract": {"contract_id": 101, "status": "WON", "profit": 9.5}}
    )
    await sl.process_contract_settlement(
        orch, {"proposal_open_contract": {"contract_id": 101, "status": "WON", "profit": 9.5}}
    )
    # untracked cid
    await sl.process_contract_settlement(
        orch, {"proposal_open_contract": {"contract_id": 999, "status": "WON", "profit": 1.0}}
    )

    # process_late_settlement_from_payload
    orch.risk_manager.register_open_contract(102, "R_10", 10.0)
    await sl.process_late_settlement_from_payload(orch, {"status": "OPEN"})
    await sl.process_late_settlement_from_payload(orch, {"status": "WON"})
    await sl.process_late_settlement_from_payload(orch, {"contract_id": 102, "status": "WON", "profit": 9.5})
    await sl.process_late_settlement_from_payload(orch, {"contract_id": 102, "status": "WON", "profit": 9.5})
    await sl.process_late_settlement_from_payload(orch, {"contract_id": 888, "status": "WON", "profit": 1.0})


@pytest.mark.asyncio
async def test_settlement_queue_ops_and_post_settlement_cycle():
    orch = _orch()
    fake_client = MagicMock(
        zremrangebyscore=AsyncMock(),
        zadd=AsyncMock(),
        pipeline=MagicMock(return_value=MagicMock(delete=MagicMock(), execute=AsyncMock())),
    )
    with patch(
        "src.application.services.orchestrator.settlement_queue_ops.aioredis.from_url", return_value=fake_client
    ):
        client = await sqo.get_redis_client(orch)
        assert client is fake_client
        await sqo.push_to_redis_priority_queue(orch, {"proposal_open_contract": {"contract_id": "bad"}})
        await sqo.push_to_redis_priority_queue(orch, {"proposal_open_contract": {"contract_id": 501}})
        await sqo.push_to_redis_priority_queue(orch, {"proposal_open_contract": {"contract_id": 501}})

        fake_client.zadd = AsyncMock(side_effect=RuntimeError("redis err"))
        await sqo.push_to_redis_priority_queue(orch, {"proposal_open_contract": {"contract_id": 502}})

    # post_settlement_cycle helpers
    assert psc.post_settlement_cycle_pending(orch) is False
    await psc._poll_delay(0.0)
    await psc._await_post_settlement_breath(orch, 0.0, 0.01)

    orch._post_settlement_wake.set()
    await psc._await_post_settlement_breath(orch, 0.05, 0.01)

    orch.is_trading = True
    await psc._release_stuck_trading_slot(orch, wait_limit=0.0)
    assert orch.is_trading is False

    orch.risk_manager.register_open_contract(777, "R_10", 5.0)
    psc._prune_stale_risk_contract_ids(orch)
    assert orch.risk_manager.active_contract_ids == []

    with patch(
        "src.application.services.orchestrator.post_settlement_cycle.get_redis_client",
        AsyncMock(return_value=fake_client),
    ):
        await psc._clean_stale_settlement_and_redis_counters(orch)
    with patch(
        "src.application.services.orchestrator.post_settlement_cycle.get_redis_client",
        AsyncMock(side_effect=RuntimeError("r")),
    ):
        await psc._clean_stale_settlement_and_redis_counters(orch)

    assert await psc._attempt_post_settlement_trading_cycle(orch, {}) is True
    orch._run_trading_cycle_if_ready = AsyncMock(side_effect=TimeoutError())
    assert (
        await psc._attempt_post_settlement_trading_cycle(orch, {"post_settlement_cycle_timeout_seconds": 0.01}) is False
    )

    await psc._await_exec_empty_signature_alignment(orch, 0.0)
    orch.risk_manager.pending_loss = {"R_10": 5.0}
    orch._cooldown_until = 9999999999.0
    with patch(
        "src.application.services.orchestrator.post_settlement_cycle._await_post_settlement_breath", AsyncMock()
    ):
        await psc._await_exec_empty_signature_alignment(orch, 0.0)
        orch._cooldown_until = 0.0
        with patch(
            "src.application.services.orchestrator.post_settlement_cycle.seconds_until_next_signature_boundary",
            return_value=10.0,
        ):
            await psc._await_exec_empty_signature_alignment(orch, 0.0)

    with patch("src.application.services.orchestrator.post_settlement_cycle.SettlementOrphanCleaner") as cleaner:
        cleaner.return_value.reconcile_stale_contracts = AsyncMock(return_value=1)
        assert await psc._apply_tolerance_window_recovery(orch, window=60.0, failed_attempts=1) == 1

    # _run_post_settlement_retry_loop and run_post_settlement_breath_and_cycle
    orch._run_trading_cycle_if_ready = AsyncMock()
    orch._last_cycle_cluster_executed = True
    await psc.run_post_settlement_breath_and_cycle(orch)

    # schedule_trading_cycle_after_settlement
    with patch(
        "src.application.services.orchestrator.post_settlement_cycle.spawn_background",
        return_value=MagicMock(done=lambda: True, add_done_callback=MagicMock()),
    ) as sb:
        psc.schedule_trading_cycle_after_settlement(orch)
        sb.assert_called_once()
