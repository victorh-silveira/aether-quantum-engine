import asyncio
from unittest.mock import patch

import pytest

from src.infrastructure.inference import meta_classifier_pool as pool
from src.infrastructure.inference.meta_classifier_client import build_meta_predict_request
from src.infrastructure.inference.meta_classifier_pool import (
    bootstrap_meta_classifier_client,
    close_meta_classifier_client,
    get_meta_classifier_client,
    predict_meta_via_config_sync,
)


def _meta_metrics() -> dict:
    return {"feature_vector": [0.1] * 34}


@pytest.mark.asyncio
async def test_bootstrap_meta_classifier_client_when_enabled():
    await close_meta_classifier_client()
    cfg = {"infra": {"meta_classifier": {"enabled": True, "http_url": "http://localhost:8005"}}}
    client = await bootstrap_meta_classifier_client(cfg)
    assert client is not None
    assert client.enabled is True
    await close_meta_classifier_client()


@pytest.mark.asyncio
async def test_bootstrap_meta_classifier_client_skips_when_disabled():
    await close_meta_classifier_client()
    cfg = {"infra": {"meta_classifier": {"enabled": False}}}
    client = await bootstrap_meta_classifier_client(cfg)
    assert client is None


@pytest.mark.asyncio
async def test_get_and_close_meta_classifier_client_singleton():
    await close_meta_classifier_client()
    cfg = {"infra": {"meta_classifier": {"enabled": True, "http_url": "http://localhost:8005"}}}
    first = await get_meta_classifier_client(cfg)
    second = await get_meta_classifier_client(cfg)
    assert first is second
    await close_meta_classifier_client()


@pytest.mark.asyncio
async def test_predict_meta_via_config_sync_uses_singleton():
    await close_meta_classifier_client()
    cfg = {"infra": {"meta_classifier": {"enabled": False}}}
    response = predict_meta_via_config_sync(
        cfg,
        build_meta_predict_request(
            symbol="OTC_SPC",
            metrics=_meta_metrics(),
            tcn_probability=0.62,
            direction="CALL",
        ),
        fallback_score=0.62,
    )
    assert response["predicted_payoff_edge"] == pytest.approx(0.0)
    await close_meta_classifier_client()


def test_predict_meta_via_config_sync_outside_loop():
    cfg = {"infra": {"meta_classifier": {"enabled": False}}}
    response = predict_meta_via_config_sync(
        cfg,
        build_meta_predict_request(
            symbol="OTC_SPC",
            metrics=_meta_metrics(),
            tcn_probability=0.62,
            direction="CALL",
        ),
        fallback_score=0.62,
    )
    assert response["predicted_payoff_edge"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_predict_meta_via_config_sync_inside_running_loop():
    cfg = {"infra": {"meta_classifier": {"enabled": False}}}
    with patch(
        "src.infrastructure.inference.meta_classifier_pool.asyncio.run",
        return_value={"predicted_payoff_edge": 0.0, "meta_applied": False},
    ):
        response = predict_meta_via_config_sync(
            cfg,
            build_meta_predict_request(
                symbol="OTC_SPC",
                metrics=_meta_metrics(),
                tcn_probability=0.62,
                direction="CALL",
            ),
            fallback_score=0.62,
        )
    assert response["predicted_payoff_edge"] == pytest.approx(0.0)


def test_get_meta_classifier_client_rebinds_across_event_loops():
    async def _once():
        cfg = {"infra": {"meta_classifier": {"enabled": True, "http_url": "http://localhost:8005"}}}
        return await get_meta_classifier_client(cfg)

    asyncio.run(close_meta_classifier_client())
    first = asyncio.run(_once())
    second = asyncio.run(_once())
    assert first is not second
    assert pool._MetaClientPool.client is second
    asyncio.run(close_meta_classifier_client())
