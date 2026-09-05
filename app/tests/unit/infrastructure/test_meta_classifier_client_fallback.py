from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.infrastructure.inference.meta_classifier_client import (
    MetaClassifierClient,
    build_meta_predict_request,
    fallback_payoff_score,
)


def _meta_metrics() -> dict:
    base = [0.1] * 14
    return {
        "feature_vector": base,
        "cross_symbol_features": {
            "cross_symbol_prob_delta": 0.1,
            "cross_symbol_vol_ratio_diff": 0.0,
            "cross_symbol_rsi_spread": 0.0,
        },
    }


def test_fallback_payoff_score_prefers_trade_score():
    metrics = {"trade_score": 0.59, "calibrated_prob": 0.62}
    assert fallback_payoff_score(metrics, "CALL", 0.62) == pytest.approx(0.59)


@pytest.mark.asyncio
async def test_predict_meta_success_clears_fallback_dedupe_state(caplog):
    client = MetaClassifierClient(base_url="http://meta:8005", timeout=1.0, enabled=True)
    client._client.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
    await client.predict_meta(
        build_meta_predict_request(
            symbol="R_10",
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
                symbol="R_10",
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
