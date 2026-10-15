"""Reserva de perdas por conta e liquidacao unica."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.application.services import pullback_recovery_claim as recovery
from src.infrastructure.state.redis_state_store import RedisStateStore


def _orch():
    return SimpleNamespace(
        config={"trading": {"mode": "demo"}},
        deriv_account_id="DOT1",
        risk_manager=SimpleNamespace(pending_loss_total=lambda: 100.0, active_contract_ids=set()),
    )


@pytest.mark.asyncio
async def test_local_claim_bind_release_and_duplicate():
    orch = _orch()
    assert recovery.unreserved_loss(orch) == 100
    assert await recovery.claim_recovery(orch, 0) is None
    token = await recovery.claim_recovery(orch, 40)
    assert token
    assert recovery.unreserved_loss(orch) == 60
    assert await recovery.claim_recovery(orch, 20) is None
    recovery.bind_recovery_contract(orch, token, 7)
    assert recovery.unreserved_loss(orch) == 60
    await recovery.release_recovery(orch, token)
    assert recovery.unreserved_loss(orch) == 60
    await recovery.release_recovery(orch, 7)
    assert recovery.unreserved_loss(orch) == 100
    recovery.bind_recovery_contract(orch, "missing", 8)


@pytest.mark.asyncio
async def test_active_unclaimed_contract_does_not_double_recover():
    orch = _orch()
    orch.risk_manager.active_contract_ids = {3}
    assert recovery.unreserved_loss(orch) == 0


@pytest.mark.asyncio
async def test_unknown_purchase_claim_expires_with_quarantine(monkeypatch):
    orch = _orch()
    clock = [100.0]
    monkeypatch.setattr(recovery.time, "monotonic", lambda: clock[0])
    token = await recovery.claim_recovery(orch, 40)
    assert token
    assert recovery.unreserved_loss(orch) == 60
    clock[0] = 1001.0
    assert recovery.unreserved_loss(orch) == 100
    assert orch._pullback_recovery_claims == {}


@pytest.mark.asyncio
async def test_redis_claim_is_atomic_and_released_by_token(monkeypatch):
    orch = _orch()
    client = SimpleNamespace(set=AsyncMock(return_value=True), eval=AsyncMock(return_value=1))
    store = object.__new__(RedisStateStore)
    monkeypatch.setattr(store, "_redis", AsyncMock(return_value=client))
    monkeypatch.setattr(store, "_full_key", lambda key: key)
    orch.state_store = store
    token = await recovery.claim_recovery(orch, 20)
    assert token
    assert client.set.await_args.kwargs == {"nx": True, "ex": 900}
    await recovery.release_recovery(orch, token)
    client.eval.assert_awaited_once()


@pytest.mark.asyncio
async def test_redis_claim_rejection_and_failure(monkeypatch):
    orch = _orch()
    client = SimpleNamespace(set=AsyncMock(return_value=False))
    store = object.__new__(RedisStateStore)
    monkeypatch.setattr(store, "_redis", AsyncMock(return_value=client))
    monkeypatch.setattr(store, "_full_key", lambda key: key)
    orch.state_store = store
    assert await recovery.claim_recovery(orch, 20) is None
    client.set.side_effect = OSError("offline")
    assert await recovery.claim_recovery(orch, 20) is None
    client.set.side_effect = None
    client.set.return_value = True
    client.eval = AsyncMock(side_effect=OSError("offline"))
    token = await recovery.claim_recovery(orch, 20)
    await recovery.release_recovery(orch, token)
