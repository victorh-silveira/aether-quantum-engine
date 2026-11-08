"""Reserva atomica de compra por ativo e vela entre processos."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.application.services.indicator_consensus import (
    claim_indicator_order,
    claim_indicator_order_once,
    quarantine_unknown_purchase,
    release_unbought_order_claim,
)
from src.infrastructure.state.redis_state_store import RedisStateStore


def _orchestrator(store):
    return SimpleNamespace(
        state_store=store,
        config={"trading": {"mode": "demo"}},
        deriv_account_id="DOT123",
    )


@pytest.mark.asyncio
async def test_order_claim_uses_redis_nx_across_orchestrators():
    store = RedisStateStore(url="redis://unused")
    client = SimpleNamespace(set=AsyncMock(side_effect=[True, False]), exists=AsyncMock(return_value=False))
    store._redis = AsyncMock(return_value=client)
    metrics = {"indicator_controls_execution": True, "indicator_candle_epoch": 600}
    first = _orchestrator(store)
    second = _orchestrator(store)
    assert await asyncio.gather(
        claim_indicator_order_once(first, "R_10", metrics),
        claim_indicator_order_once(second, "R_10", metrics),
    ) == [True, False]
    assert not await claim_indicator_order_once(first, "R_10", metrics)
    assert client.set.await_count == 2
    assert client.set.await_args.kwargs == {"nx": True, "ex": 900}
    assert ":demo:DOT123:R_10:300:600" in client.set.await_args.args[0]


@pytest.mark.asyncio
async def test_multi_period_claim_reserves_symbol_across_periods():
    store = RedisStateStore(url="redis://unused")
    client = SimpleNamespace(
        set=AsyncMock(side_effect=[True, True, True, False]),
        exists=AsyncMock(return_value=False),
        eval=AsyncMock(return_value=1),
    )
    store._redis = AsyncMock(return_value=client)
    orch = _orchestrator(store)
    orch.config["indicator_consensus"] = {"multi_period_trading": True}
    first = {"indicator_controls_execution": True, "indicator_candle_epoch": 900, "indicator_period_seconds": 900}
    assert await claim_indicator_order_once(orch, "R_25", first)
    assert first["_symbol_claim_token"] == first["_order_claim_token"]
    assert client.set.await_args.kwargs == {"nx": True, "ex": 930}
    second = {"indicator_controls_execution": True, "indicator_candle_epoch": 900, "indicator_period_seconds": 300}
    assert not await claim_indicator_order_once(orch, "R_25", second)
    assert client.eval.await_count == 2
    await release_unbought_order_claim(orch, "R_25", first)
    assert client.eval.await_count == 4


@pytest.mark.asyncio
async def test_multi_period_symbol_claim_redis_failure_releases_period_claim():
    store = RedisStateStore(url="redis://unused")
    client = SimpleNamespace(
        set=AsyncMock(side_effect=[True, OSError("offline")]),
        exists=AsyncMock(return_value=False),
        eval=AsyncMock(return_value=1),
    )
    store._redis = AsyncMock(return_value=client)
    orch = _orchestrator(store)
    orch.config["indicator_consensus"] = {"multi_period_trading": True}
    metrics = {"indicator_candle_epoch": 900, "indicator_period_seconds": 900}
    assert not await claim_indicator_order_once(orch, "R_25", metrics)
    assert client.eval.await_count == 2
    assert orch._indicator_order_claims == set()


@pytest.mark.asyncio
async def test_order_claim_fails_closed_if_redis_is_unavailable():
    store = RedisStateStore(url="redis://unused")
    store._redis = AsyncMock(side_effect=OSError("offline"))
    orch = _orchestrator(store)
    assert not await claim_indicator_order_once(
        orch, "R_25", {"indicator_controls_execution": True, "indicator_candle_epoch": 600}
    )


@pytest.mark.asyncio
async def test_order_claim_fails_closed_if_unknown_purchase_lookup_fails():
    store = RedisStateStore(url="redis://unused")
    client = SimpleNamespace(exists=AsyncMock(side_effect=OSError("offline")), set=AsyncMock())
    store._redis = AsyncMock(return_value=client)
    assert not await claim_indicator_order_once(_orchestrator(store), "R_25", {})
    client.set.assert_not_awaited()


@pytest.mark.asyncio
async def test_order_claim_fails_closed_if_atomic_reservation_fails():
    store = RedisStateStore(url="redis://unused")
    client = SimpleNamespace(exists=AsyncMock(return_value=False), set=AsyncMock(side_effect=OSError("offline")))
    store._redis = AsyncMock(return_value=client)
    assert not await claim_indicator_order_once(_orchestrator(store), "R_25", {})
    client.set.assert_awaited_once()


@pytest.mark.asyncio
async def test_non_indicator_order_claims_current_candle():
    store = RedisStateStore(url="redis://unused")
    client = SimpleNamespace(set=AsyncMock(return_value=True), exists=AsyncMock(return_value=False))
    store._redis = AsyncMock(return_value=client)
    assert await claim_indicator_order_once(_orchestrator(store), "R_10", {})
    client.set.assert_awaited_once()


def test_invalid_order_epoch_is_not_reserved():
    assert not claim_indicator_order(_orchestrator(None), "R_10", {"model_input_candle_epoch": -300})


@pytest.mark.asyncio
async def test_unknown_purchase_quarantines_same_symbol_locally_and_in_redis():
    store = RedisStateStore(url="redis://unused")
    client = SimpleNamespace(set=AsyncMock(), exists=AsyncMock(return_value=True))
    store._redis = AsyncMock(return_value=client)
    orch = _orchestrator(store)
    await quarantine_unknown_purchase(orch, "R_25")
    assert client.set.await_args.kwargs == {"ex": 900}
    assert not await claim_indicator_order_once(
        orch, "R_25", {"indicator_controls_execution": True, "indicator_candle_epoch": 600}
    )
    restarted = _orchestrator(store)
    assert not await claim_indicator_order_once(restarted, "R_25", {"model_input_candle_epoch": 600})
    client.exists.assert_awaited_once()


@pytest.mark.asyncio
async def test_unknown_purchase_still_quarantines_locally_if_redis_write_fails():
    store = RedisStateStore(url="redis://unused")
    store._redis = AsyncMock(side_effect=OSError("offline"))
    orch = _orchestrator(store)
    await quarantine_unknown_purchase(orch, "R_10")
    assert not await claim_indicator_order_once(orch, "R_10", {})


@pytest.mark.asyncio
async def test_rejected_proposal_releases_only_owned_redis_claim():
    store = RedisStateStore(url="redis://unused")
    client = SimpleNamespace(
        exists=AsyncMock(return_value=False),
        set=AsyncMock(return_value=True),
        eval=AsyncMock(return_value=1),
    )
    store._redis = AsyncMock(return_value=client)
    orch = _orchestrator(store)
    metrics = {"indicator_controls_execution": True, "indicator_candle_epoch": 600}
    assert await claim_indicator_order_once(orch, "R_25", metrics)
    token = metrics["_order_claim_token"]
    assert client.set.await_args.args[1] == token
    await release_unbought_order_claim(orch, "R_25", metrics)
    assert client.eval.await_args.args[3] == token
    assert orch._indicator_order_claims == set()
    assert await claim_indicator_order_once(orch, "R_25", {"indicator_candle_epoch": 600})


@pytest.mark.asyncio
async def test_failed_redis_release_preserves_local_claim():
    store = RedisStateStore(url="redis://unused")
    client = SimpleNamespace(
        exists=AsyncMock(return_value=False),
        set=AsyncMock(return_value=True),
        eval=AsyncMock(side_effect=OSError("offline")),
    )
    store._redis = AsyncMock(return_value=client)
    orch = _orchestrator(store)
    metrics = {"indicator_candle_epoch": 600}
    assert await claim_indicator_order_once(orch, "R_25", metrics)
    await release_unbought_order_claim(orch, "R_25", metrics)
    assert ("R_25", 300, 600) in orch._indicator_order_claims


@pytest.mark.asyncio
async def test_local_claim_without_redis_can_be_released():
    orch = _orchestrator(None)
    metrics = {"indicator_candle_epoch": 600}
    assert await claim_indicator_order_once(orch, "R_25", metrics)
    await release_unbought_order_claim(orch, "R_25", metrics)
    assert orch._indicator_order_claims == set()


@pytest.mark.asyncio
async def test_multi_period_residual_claim_overridden_when_no_active_contracts():
    from src.application.services.indicator_order_claims import (
        release_order_symbol_claim,
    )

    store = RedisStateStore(url="redis://unused")
    client = SimpleNamespace(
        exists=AsyncMock(return_value=False),
        set=AsyncMock(side_effect=[True, False, True]),
        delete=AsyncMock(return_value=1),
    )
    store._redis = AsyncMock(return_value=client)
    orch = _orchestrator(store)
    orch.config["indicator_consensus"] = {"multi_period_trading": True}
    orch.state = SimpleNamespace(active_contracts={})
    orch.risk_manager = SimpleNamespace(contract_to_symbol={}, active_contract_ids=set())

    metrics = {"indicator_controls_execution": True, "indicator_candle_epoch": 1200, "indicator_period_seconds": 900}
    assert await claim_indicator_order_once(orch, "OTC_SSMI", metrics)
    assert metrics.get("_symbol_claim_token") is not None
    assert client.set.await_count == 3

    await release_order_symbol_claim(orch, "OTC_SSMI")
    client.delete.assert_awaited_once()

    client.delete.side_effect = OSError("redis error")
    await release_order_symbol_claim(orch, "OTC_SSMI")


@pytest.mark.asyncio
async def test_multi_period_claim_blocked_when_active_contract_exists():
    store = RedisStateStore(url="redis://unused")
    client = SimpleNamespace(
        exists=AsyncMock(return_value=False),
        set=AsyncMock(side_effect=[True, False]),
        eval=AsyncMock(return_value=1),
    )
    store._redis = AsyncMock(return_value=client)
    orch = _orchestrator(store)
    orch.config["indicator_consensus"] = {"multi_period_trading": True}
    orch.state = SimpleNamespace(active_contracts={101: SimpleNamespace(symbol="OTC_SSMI")})

    metrics = {"indicator_controls_execution": True, "indicator_candle_epoch": 1200, "indicator_period_seconds": 900}
    assert not await claim_indicator_order_once(orch, "OTC_SSMI", metrics)
    assert client.eval.await_count == 2


def test_has_active_contract_for_symbol_state_and_risk():
    from src.application.services.indicator_order_claims import (
        _has_active_contract_for_symbol,
        claim_indicator_order,
    )

    orch = _orchestrator(None)
    assert not claim_indicator_order(orch, "R_25", {"indicator_controls_execution": True})
    orch.state = SimpleNamespace(active_contracts={1: SimpleNamespace(symbol="OTC_SSMI")})
    assert _has_active_contract_for_symbol(orch, "OTC_SSMI")
    assert not _has_active_contract_for_symbol(orch, "OTC_NDX")

    orch.state = SimpleNamespace(active_contracts={})
    orch.risk_manager = SimpleNamespace(contract_to_symbol={2: "OTC_NDX"}, active_contract_ids={2})
    assert _has_active_contract_for_symbol(orch, "OTC_NDX")
    assert not _has_active_contract_for_symbol(orch, "OTC_SSMI")
