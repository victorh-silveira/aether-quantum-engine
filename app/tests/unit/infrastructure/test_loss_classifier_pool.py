import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.infrastructure.inference import loss_classifier_pool as pool
from src.infrastructure.inference.loss_classifier_pool import (
    close_loss_classifier_client,
    get_loss_classifier_client,
    learn_loss_via_config_sync,
    predict_loss_via_config_sync,
)


def _cfg(*, enabled: bool = False) -> dict:
    return {"infra": {"loss_classifier": {"enabled": enabled, "http_url": "http://localhost:8006"}}}


def _predict_request() -> dict:
    return {
        "feature_vector": [0.1, 0.2],
        "symbol": "1HZ75V",
        "direction": "CALL",
        "veto_p_loss_floor": 0.9,
    }


@pytest.mark.asyncio
async def test_get_and_close_loss_classifier_client_singleton():
    await close_loss_classifier_client()
    first = await get_loss_classifier_client(_cfg())
    second = await get_loss_classifier_client(_cfg())
    assert first is second
    await close_loss_classifier_client()
    assert pool._LossClientPool.client is None


@pytest.mark.asyncio
async def test_close_loss_classifier_client_clears_pool():
    await get_loss_classifier_client(_cfg())
    await close_loss_classifier_client()
    assert pool._LossClientPool.client is None
    assert pool._LossClientPool.loop is None


def test_get_loss_classifier_client_rebinds_across_event_loops():
    async def _once():
        return await get_loss_classifier_client(_cfg(enabled=True))

    asyncio.run(close_loss_classifier_client())
    first = asyncio.run(_once())
    second = asyncio.run(_once())
    assert first is not second
    assert pool._LossClientPool.client is second
    asyncio.run(close_loss_classifier_client())


def test_predict_loss_via_config_sync_outside_loop():
    asyncio.run(close_loss_classifier_client())
    result = predict_loss_via_config_sync(_cfg(enabled=False), _predict_request())
    assert result["p_loss"] == pytest.approx(0.5)
    assert result["veto"] is False
    asyncio.run(close_loss_classifier_client())


@pytest.mark.asyncio
async def test_predict_loss_via_config_sync_inside_running_loop():
    await close_loss_classifier_client()
    with patch(
        "src.infrastructure.inference.loss_classifier_pool.asyncio.run",
        return_value={
            "p_loss": 0.5,
            "veto": False,
            "auto_learn_applied": False,
            "model_version": "none",
            "n_train": 0,
            "veto_ready": False,
            "bootstrap": False,
            "collapsed": False,
        },
    ):
        result = predict_loss_via_config_sync(_cfg(enabled=False), _predict_request())
    assert result["p_loss"] == pytest.approx(0.5)
    await close_loss_classifier_client()


def test_learn_loss_via_config_sync_disabled():
    asyncio.run(close_loss_classifier_client())
    result = learn_loss_via_config_sync(_cfg(enabled=False), feature_vector=[0.1], label="LOSS")
    assert result == {"ok": False, "skipped": True}
    asyncio.run(close_loss_classifier_client())


def test_learn_loss_via_config_sync_outside_loop():
    asyncio.run(close_loss_classifier_client())
    result = learn_loss_via_config_sync(_cfg(enabled=False), feature_vector=[0.1], label="WIN")
    assert result["skipped"] is True
    asyncio.run(close_loss_classifier_client())


@pytest.mark.asyncio
async def test_learn_loss_via_config_sync_inside_running_loop():
    await close_loss_classifier_client()
    with patch(
        "src.infrastructure.inference.loss_classifier_pool.asyncio.run",
        return_value={"ok": False, "skipped": True},
    ):
        result = learn_loss_via_config_sync(_cfg(enabled=False), feature_vector=[0.1], label="LOSS")
    assert result["skipped"] is True
    await close_loss_classifier_client()


@pytest.mark.asyncio
async def test_learn_loss_via_config_sync_enabled_calls_client():
    await close_loss_classifier_client()
    mock_client = MagicMock()
    mock_client.learn = AsyncMock(return_value={"ok": True, "n_train": 3})
    with patch(
        "src.infrastructure.inference.loss_classifier_pool.get_loss_classifier_client",
        new=AsyncMock(return_value=mock_client),
    ):
        result = learn_loss_via_config_sync(
            _cfg(enabled=True),
            feature_vector=[0.1],
            label="LOSS",
            contract_id="c1",
            symbol="1HZ75V",
        )
    assert result["ok"] is True
    mock_client.learn.assert_awaited_once()
    await close_loss_classifier_client()

    await close_loss_classifier_client()
    stale = MagicMock()
    stale.aclose = AsyncMock()
    pool._LossClientPool.client = stale
    pool._LossClientPool.loop = asyncio.new_event_loop()
    client = await get_loss_classifier_client(_cfg())
    stale.aclose.assert_awaited_once()
    assert client is not stale
    await close_loss_classifier_client()
