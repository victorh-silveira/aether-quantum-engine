from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.infrastructure.inference.meta_classifier_client import (
    MetaClassifierClient,
    build_meta_predict_request,
    build_persistent_http_client,
    fallback_payoff_score,
    meta_classifier_enabled,
    meta_classifier_http_url,
    meta_classifier_timeout,
    reset_meta_classifier_fallback_dedupe,
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
    response.json = MagicMock(return_value={"predicted_payoff_edge": 0.17, "meta_applied": True})
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
    assert result["predicted_payoff_edge"] == pytest.approx(0.17)
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
    assert result["predicted_payoff_edge"] == pytest.approx(0.0)
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
    assert result["predicted_payoff_edge"] == pytest.approx(0.0)
    assert result["meta_applied"] is False
    await client.aclose()


@pytest.mark.asyncio
async def test_predict_meta_disabled_returns_zero_edge():
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
    assert result["predicted_payoff_edge"] == pytest.approx(0.0)
    assert result["meta_applied"] is False
    await client.aclose()


@pytest.mark.asyncio
async def test_predict_meta_batch_emits_single_fallback_log(caplog):
    client = MetaClassifierClient(base_url="http://meta:8005", timeout=1.0, enabled=True)
    client._client.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
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
    with caplog.at_level("WARNING"):
        results = await client.predict_meta_batch(requests)
    assert len(results) == 2
    fallback_logs = [r for r in caplog.records if "META_CLASSIFIER_FALLBACK" in r.message]
    assert len(fallback_logs) == 1
    await client.aclose()


def test_build_persistent_http_client_uses_keepalive_limits():
    http_client = build_persistent_http_client("http://meta:8005", 1.0)
    assert http_client.base_url == "http://meta:8005"
    assert http_client.timeout.connect == pytest.approx(1.0)


def test_reset_meta_classifier_fallback_dedupe_clears_channel():
    reset_meta_classifier_fallback_dedupe()
    reset_meta_classifier_fallback_dedupe()


@pytest.mark.asyncio
async def test_meta_classifier_client_accepts_injected_http_client():
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.aclose = AsyncMock()
    client = MetaClassifierClient(
        base_url="http://meta:8005",
        timeout=1.0,
        enabled=False,
        http_client=mock_client,
        owns_http_client=False,
    )
    assert client._client is mock_client
    await client.aclose()
    mock_client.aclose.assert_not_awaited()


@pytest.mark.asyncio
async def test_predict_meta_batch_parallel():
    client = MetaClassifierClient(base_url="http://meta:8005", timeout=1.0, enabled=True)
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json = MagicMock(return_value={"predicted_payoff_edge": 0.11, "meta_applied": True})
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
    assert results[0]["predicted_payoff_edge"] == pytest.approx(0.11)
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
    assert result["predicted_payoff_edge"] == pytest.approx(0.0)


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
    assert result["predicted_payoff_edge"] == pytest.approx(0.0)


def test_fallback_payoff_score_prefers_trade_score():
    metrics = {"trade_score": 0.59, "calibrated_prob": 0.62}
    assert fallback_payoff_score(metrics, "CALL", 0.62) == pytest.approx(0.59)


@pytest.mark.asyncio
async def test_predict_meta_success_clears_fallback_dedupe_state(caplog):
    client = MetaClassifierClient(base_url="http://meta:8005", timeout=1.0, enabled=True)
    client._client.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
    await client.predict_meta(
        build_meta_predict_request(
            symbol="RDBULL",
            metrics=_meta_metrics(),
            tcn_probability=0.62,
            direction="CALL",
        ),
        fallback_score=0.62,
    )
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json = MagicMock(return_value={"predicted_payoff_edge": 0.17, "meta_applied": True})
    client._client.post = AsyncMock(return_value=response)
    with caplog.at_level("WARNING"):
        result = await client.predict_meta(
            build_meta_predict_request(
                symbol="RDBULL",
                metrics=_meta_metrics(),
                tcn_probability=0.62,
                direction="CALL",
            ),
            fallback_score=0.62,
        )
    assert result["meta_applied"] is True
    fallback_logs = [record for record in caplog.records if "META_CLASSIFIER_FALLBACK" in record.message]
    assert len(fallback_logs) == 1
    await client.aclose()
