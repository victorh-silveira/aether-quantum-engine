"""Testes do JsonStateStore."""

import pytest

from src.infrastructure.state.json_state_store import JsonStateStore


@pytest.mark.asyncio
async def test_json_state_store_session_keys(tmp_path):
    path = tmp_path / "state.json"
    store = JsonStateStore(path)
    await store.save_state_bundle(
        snapshot={"risk": {"consecutive_losses_linear": 0}},
        session_start_balance=2000.0,
        session_target_win=20.0,
        dlambert_unit=15.0,
        consecutive_losses_linear=1,
    )
    assert await store.get_string("session:current:start_balance") == "2000.0"
    assert await store.get_string("session:current:target_win") == "20.0"
    assert await store.get_string("session:current:dlambert_unit") == "15.0"
    assert await store.get_string("session:current:consecutive_losses_linear") == "1"


@pytest.mark.asyncio
async def test_json_state_store_roundtrip(tmp_path):
    path = tmp_path / "state.json"
    store = JsonStateStore(path)
    await store.save_snapshot({"risk": {"pending_loss": {"R_10": 1.0}}})
    loaded = await store.load_snapshot()
    assert loaded["risk"]["pending_loss"]["R_10"] == 1.0
    await store.set_string("market_sig", "sig")
    assert await store.get_string("market_sig") == "sig"
    assert await store.incr_string("recovery:skip_counter") == 1
    assert await store.incr_string("recovery:skip_counter") == 2
    await store.set_string("recovery:skip_counter", "bad")
    assert await store.incr_string("recovery:skip_counter") == 1
    store.save({"sync": True})
    assert store.load()["sync"] is True
