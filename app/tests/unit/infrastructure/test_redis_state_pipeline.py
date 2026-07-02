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
        snapshot={"risk": {"consecutive_losses_linear": 1, "pending_loss": {"RDBEAR": 4.0}}},
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
        snapshot={"risk": {"consecutive_losses_linear": 0, "pending_loss": {}}},
    )
    pipe.delete.assert_called()


@pytest.mark.asyncio
async def test_write_state_bundle_includes_recovery_skip_counter():
    client = MagicMock()
    pipe = _pipeline_ctx()
    client.pipeline.return_value = pipe
    await write_state_bundle(
        client,
        prefix="aether",
        snapshot={"risk": {"consecutive_losses_linear": 2}},
        recovery_skip_counter=3,
    )
    pipe.set.assert_called()
    set_args = [call.args[0] for call in pipe.set.call_args_list]
    assert any("recovery:skip_counter" in str(key) for key in set_args)


@pytest.mark.asyncio
async def test_write_state_bundle_includes_dlambert_keys():
    client = MagicMock()
    pipe = _pipeline_ctx()
    client.pipeline.return_value = pipe
    await write_state_bundle(
        client,
        prefix="aether",
        snapshot={"risk": {"consecutive_losses_linear": 1}},
        dlambert_unit=25.5,
        consecutive_losses_linear=2,
    )
    set_args = [call.args[0] for call in pipe.set.call_args_list]
    assert any("session:current:dlambert_unit" in str(key) for key in set_args)
    assert any("session:current:consecutive_losses_linear" in str(key) for key in set_args)


@pytest.mark.asyncio
async def test_write_state_bundle_includes_session_current_keys():
    client = MagicMock()
    pipe = _pipeline_ctx()
    client.pipeline.return_value = pipe
    await write_state_bundle(
        client,
        prefix="aether",
        snapshot={"risk": {"consecutive_losses_linear": 0}},
        session_start_balance=10000.0,
        session_target_win=100.0,
    )
    set_args = [call.args[0] for call in pipe.set.call_args_list]
    assert any("session:current:start_balance" in str(key) for key in set_args)
    assert any("session:current:target_win" in str(key) for key in set_args)
