"""Testes do pipeline atomico Redis."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.infrastructure.state.redis_state_pipeline import write_state_bundle


def _pipeline_ctx():
    pipe = MagicMock()
    pipe.__aenter__ = AsyncMock(return_value=pipe)
    pipe.__aexit__ = AsyncMock(return_value=None)
    pipe.execute = AsyncMock()
    return pipe


@pytest.mark.asyncio
async def test_write_state_bundle_transaction():
    client = MagicMock()
    pipe = _pipeline_ctx()
    client.pipeline.return_value = pipe
    await write_state_bundle(
        client,
        prefix="aether",
        snapshot={"risk": {"consecutive_losses": 1, "pending_loss": {"R_25": 4.0}}},
        session_hash={"day_key": 2},
        market_sig="mkt",
    )
    client.pipeline.assert_called_with(transaction=True)
    pipe.set.assert_called()
    pipe.hset.assert_called()
    pipe.delete.assert_called()
    pipe.execute.assert_awaited()


@pytest.mark.asyncio
async def test_write_state_bundle_clears_empty_pending_loss():
    client = MagicMock()
    pipe = _pipeline_ctx()
    client.pipeline.return_value = pipe
    await write_state_bundle(
        client,
        prefix="aether",
        snapshot={"risk": {"consecutive_losses": 0, "pending_loss": {}}},
    )
    pipe.delete.assert_called()
