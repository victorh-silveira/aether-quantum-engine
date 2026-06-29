"""Testes do JsonStateStore."""

import pytest

from src.infrastructure.state.json_state_store import JsonStateStore


@pytest.mark.asyncio
async def test_json_state_store_roundtrip(tmp_path):
    path = tmp_path / "state.json"
    store = JsonStateStore(path)
    await store.save_snapshot({"risk": {"pending_loss": {"R_10": 1.0}}})
    loaded = await store.load_snapshot()
    assert loaded["risk"]["pending_loss"]["R_10"] == 1.0
    await store.set_string("market_sig", "sig")
    assert await store.get_string("market_sig") == "sig"
    store.save({"sync": True})
    assert store.load()["sync"] is True
