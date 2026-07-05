from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.infrastructure.inference.meta_classifier_client import (
    MetaClassifierClient,
    build_meta_predict_request,
    close_meta_classifier_client,
    fallback_payoff_score,
    get_meta_classifier_client,
    meta_classifier_enabled,
    meta_classifier_http_url,
    meta_classifier_timeout,
    predict_meta_via_config_sync,
)


def _meta_metrics() -> dict:
    base = [0.1] * 34
    return {
        "feature_vector": base,
        "cross_symbol_features": {
            "cross_symbol_prob_delta": 0.1,
            "cross_symbol_vol_ratio_diff": 0.0,
            "cross_symbol_rsi_spread": 0.0,
        },
    }


def test_meta_classifier_enabled_from_config():
    cfg = {"infra": {"meta_classifier": {"enabled": True}}}
    assert meta_classifier_enabled(cfg) is True
    assert meta_classifier_http_url(cfg) == "http://localhost:8005"
    assert meta_classifier_timeout(cfg) == 1.0


@pytest.mark.asyncio
async def test_predict_meta_success():
    client = MetaClassifierClient(base_url="http://meta:8005", timeout=1.0, enabled=True)
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json = MagicMock(return_value={"calibrated_payoff_score": 0.77, "meta_applied": True})
    client._client.post = AsyncMock(return_value=response)
    result = await client.predict_meta(
        build_meta_predict_request(
            symbol="RDBULL",
            metrics=_meta_metrics(),
            tcn_probability=0.62,
            direction="CALL",
        ),
        fallback_score=0.62,
    )
    assert result["calibrated_payoff_score"] == pytest.approx(0.77)
    assert result["meta_applied"] is True
    await client.aclose()


@pytest.mark.asyncio
async def test_predict_meta_timeout_fallback():
    client = MetaClassifierClient(base_url="http://meta:8005", timeout=1.0, enabled=True)
    client._client.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
    result = await client.predict_meta(
        build_meta_predict_request(
            symbol="RDBULL",
            metrics=_meta_metrics(),
            tcn_probability=0.62,
            direction="CALL",
        ),
        fallback_score=0.62,
    )
    assert result["calibrated_payoff_score"] == pytest.approx(0.62)
    assert result["meta_applied"] is False
    await client.aclose()


@pytest.mark.asyncio
async def test_predict_meta_http_error_fallback():
    client = MetaClassifierClient(base_url="http://meta:8005", timeout=1.0, enabled=True)
    request = httpx.Request("POST", "http://meta:8005/v2/predict_meta")
    response = httpx.Response(503, request=request)
    client._client.post = AsyncMock(side_effect=httpx.HTTPStatusError("fail", request=request, response=response))
    result = await client.predict_meta(
        build_meta_predict_request(
            symbol="RDBULL",
            metrics=_meta_metrics(),
            tcn_probability=0.62,
            direction="CALL",
        ),
        fallback_score=0.62,
    )
    assert result["calibrated_payoff_score"] == pytest.approx(0.62)
    assert result["meta_applied"] is False
    await client.aclose()


@pytest.mark.asyncio
async def test_predict_meta_disabled_returns_fallback():
    client = MetaClassifierClient(base_url="http://meta:8005", timeout=1.0, enabled=False)
    result = await client.predict_meta(
        build_meta_predict_request(
            symbol="RDBULL",
            metrics=_meta_metrics(),
            tcn_probability=0.62,
            direction="CALL",
        ),
        fallback_score=0.62,
    )
    assert result["calibrated_payoff_score"] == pytest.approx(0.62)
    await client.aclose()


@pytest.mark.asyncio
async def test_get_and_close_meta_classifier_client_singleton():
    await close_meta_classifier_client()
    cfg = {"infra": {"meta_classifier": {"enabled": True, "http_url": "http://localhost:8005"}}}
    first = await get_meta_classifier_client(cfg)
    second = await get_meta_classifier_client(cfg)
    assert first is second
    await close_meta_classifier_client()


@pytest.mark.asyncio
async def test_predict_meta_batch_parallel():
    client = MetaClassifierClient(base_url="http://meta:8005", timeout=1.0, enabled=True)
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json = MagicMock(return_value={"calibrated_payoff_score": 0.66, "meta_applied": True})
    client._client.post = AsyncMock(return_value=response)
    requests = [
        (
            build_meta_predict_request(
                symbol="RDBULL",
                metrics=_meta_metrics(),
                tcn_probability=0.62,
                direction="CALL",
            ),
            0.62,
        ),
        (
            build_meta_predict_request(
                symbol="RDBEAR",
                metrics={"feature_vector": [0.2] * 34},
                tcn_probability=0.41,
                direction="PUT",
            ),
            0.59,
        ),
    ]
    results = await client.predict_meta_batch(requests)
    assert len(results) == 2
    assert results[0]["calibrated_payoff_score"] == pytest.approx(0.66)
    await client.aclose()


def test_predict_meta_sync_outside_running_loop():
    client = MetaClassifierClient(base_url="http://meta:8005", timeout=1.0, enabled=False)
    result = client.predict_meta_sync(
        build_meta_predict_request(
            symbol="RDBULL",
            metrics=_meta_metrics(),
            tcn_probability=0.62,
            direction="CALL",
        ),
        fallback_score=0.62,
    )
    assert result["calibrated_payoff_score"] == pytest.approx(0.62)


@pytest.mark.asyncio
async def test_predict_meta_invalid_json_fallback():
    client = MetaClassifierClient(base_url="http://meta:8005", timeout=1.0, enabled=True)
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json = MagicMock(side_effect=ValueError("bad json"))
    client._client.post = AsyncMock(return_value=response)
    result = await client.predict_meta(
        build_meta_predict_request(
            symbol="RDBULL",
            metrics=_meta_metrics(),
            tcn_probability=0.62,
            direction="CALL",
        ),
        fallback_score=0.62,
    )
    assert result["meta_applied"] is False
    await client.aclose()


def test_predict_meta_via_config_sync_outside_loop():
    cfg = {"infra": {"meta_classifier": {"enabled": False}}}
    response = predict_meta_via_config_sync(
        cfg,
        build_meta_predict_request(
            symbol="RDBULL",
            metrics=_meta_metrics(),
            tcn_probability=0.62,
            direction="CALL",
        ),
        fallback_score=0.62,
    )
    assert response["calibrated_payoff_score"] == pytest.approx(0.62)


@pytest.mark.asyncio
async def test_predict_meta_via_config_sync_inside_running_loop():
    cfg = {"infra": {"meta_classifier": {"enabled": False}}}
    with patch(
        "src.infrastructure.inference.meta_classifier_client.asyncio.run",
        return_value={"calibrated_payoff_score": 0.71, "meta_applied": False},
    ):
        response = predict_meta_via_config_sync(
            cfg,
            build_meta_predict_request(
                symbol="RDBULL",
                metrics=_meta_metrics(),
                tcn_probability=0.62,
                direction="CALL",
            ),
            fallback_score=0.62,
        )
    assert response["calibrated_payoff_score"] == pytest.approx(0.71)


def test_meta_classifier_timeout_custom_value():
    cfg = {"infra": {"meta_classifier": {"enabled": True, "timeout_seconds": 0.75}}}
    assert meta_classifier_timeout(cfg) == pytest.approx(0.75)


def test_meta_classifier_client_enabled_property():
    client = MetaClassifierClient(base_url="http://meta:8005", timeout=1.0, enabled=True)
    assert client.enabled is True


@pytest.mark.asyncio
async def test_predict_meta_sync_inside_running_loop():
    client = MetaClassifierClient(base_url="http://meta:8005", timeout=1.0, enabled=False)
    result = client.predict_meta_sync(
        build_meta_predict_request(
            symbol="RDBULL",
            metrics=_meta_metrics(),
            tcn_probability=0.62,
            direction="CALL",
        ),
        fallback_score=0.62,
    )
    assert result["calibrated_payoff_score"] == pytest.approx(0.62)


def test_fallback_payoff_score_prefers_trade_score():
    metrics = {"trade_score": 0.59, "calibrated_prob": 0.62}
    assert fallback_payoff_score(metrics, "CALL", 0.62) == pytest.approx(0.59)
