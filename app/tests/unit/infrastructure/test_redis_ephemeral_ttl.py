"""TTL Redis efemero vs settlement soberano."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.application.services.orchestrator.orchestrator_state_restore import (
    mark_bar_processed,
    sync_market_signature,
)
from src.infrastructure.state.redis_ephemeral_ttl import REDIS_EPHEMERAL_SIG_TTL_SECONDS
from src.infrastructure.state.redis_state_pipeline import write_state_bundle


@pytest.mark.asyncio
async def test_mark_bar_processed_uses_ephemeral_ttl():
    orch = MagicMock()
    orch.state_store = AsyncMock()
    await mark_bar_processed(orch, "1HZ75V", 100)
    orch.state_store.set_string.assert_awaited_once_with(
        "bar_sig:1HZ75V",
        "100",
        ttl_seconds=REDIS_EPHEMERAL_SIG_TTL_SECONDS,
    )


@pytest.mark.asyncio
async def test_sync_market_signature_uses_ephemeral_ttl():
    orch = MagicMock()
    orch.state_store = AsyncMock()
    await sync_market_signature(orch, "sig-abc")
    orch.state_store.set_string.assert_awaited_once_with(
        "market_sig",
        "sig-abc",
        ttl_seconds=REDIS_EPHEMERAL_SIG_TTL_SECONDS,
    )


@pytest.mark.asyncio
async def test_write_state_bundle_market_sig_sets_ex():
    client = MagicMock()
    pipe = MagicMock()
    pipe.__aenter__ = AsyncMock(return_value=pipe)
    pipe.__aexit__ = AsyncMock(return_value=None)
    pipe.execute = AsyncMock()
    client.pipeline.return_value = pipe
    await write_state_bundle(
        client,
        prefix="aether",
        snapshot={"risk": {}},
        market_sig="mkt",
    )
    market_calls = [c for c in pipe.set.call_args_list if len(c.args) >= 2 and c.args[1] == "mkt"]
    assert market_calls
    assert market_calls[0].kwargs.get("ex") == REDIS_EPHEMERAL_SIG_TTL_SECONDS


def test_ephemeral_ttl_seconds_is_three_m5_cycles():
    assert REDIS_EPHEMERAL_SIG_TTL_SECONDS == 900


def test_settlement_queue_ops_source_has_no_expire():
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "application"
        / "services"
        / "orchestrator"
        / "settlement_queue_ops.py"
    )
    text = src.read_text(encoding="utf-8")
    assert "settlement:queue:priority" in text
    assert ".expire(" not in text
    assert "EXPIRE" not in text
    assert "xadd" not in text.lower()
