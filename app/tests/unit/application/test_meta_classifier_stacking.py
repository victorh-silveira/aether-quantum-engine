from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.meta_classifier_features import (
    META_FEATURE_DIM,
    extract_meta_feature_vector,
    side_payoff_from_probability,
)
from src.application.services.meta_classifier_stacking import (
    apply_meta_payoff_to_metrics,
    prefetch_meta_payoff_for_decisions,
    resolve_meta_payoff_score,
)
from src.domain.models.trade import TradeDirection


def _metrics_with_cross() -> dict:
    base = [0.1] * 34
    cross = {"cross_symbol_prob_delta": 0.21, "cross_symbol_vol_ratio_diff": 0.08, "cross_symbol_rsi_spread": 12.0}
    flow = {"micro_tick_acceleration": 0.04, "keltner_deviation_ratio": -0.11}
    return {
        "calibrated_prob": 0.62,
        "feature_vector": base,
        "cross_symbol_features": cross,
        "flow_features": flow,
        "meta_feature_vector": base + [0.21, 0.08, 12.0, 0.04, -0.11],
    }


def test_extract_meta_feature_vector_expanded_with_cross_symbol():
    vector = extract_meta_feature_vector(_metrics_with_cross())
    assert len(vector) == META_FEATURE_DIM
    assert vector[-5:] == pytest.approx([0.21, 0.08, 12.0, 0.04, -0.11])


def test_side_payoff_from_probability_put():
    assert side_payoff_from_probability(0.62, "PUT") == pytest.approx(0.38)


@pytest.mark.asyncio
async def test_prefetch_meta_payoff_for_decisions_with_cross_symbol_payload():
    decisions = {
        "RDBULL": {"direction": TradeDirection.CALL, "metrics": _metrics_with_cross()},
        "RDBEAR": {
            "direction": TradeDirection.PUT,
            "metrics": {
                "calibrated_prob": 0.39,
                "feature_vector": [0.2] * 34,
                "micro_indicators": {"rsi": 50.0, "vol_ratio": 1.0},
            },
        },
    }
    cfg = {"infra": {"meta_classifier": {"enabled": True, "http_url": "http://localhost:8005"}}}
    with patch(
        "src.application.services.meta_classifier_stacking.get_meta_classifier_client",
        new_callable=AsyncMock,
    ) as get_client:
        client = MagicMock()
        client.predict_meta_batch = AsyncMock(
            return_value=[
                {"calibrated_payoff_score": 0.71, "meta_applied": True},
                {"calibrated_payoff_score": 0.59, "meta_applied": True},
            ]
        )
        get_client.return_value = client
        await prefetch_meta_payoff_for_decisions(decisions, cfg)
    bull = decisions["RDBULL"]["metrics"]
    bear = decisions["RDBEAR"]["metrics"]
    assert bull["trade_score"] == pytest.approx(0.71)
    assert bear["trade_score"] == pytest.approx(0.59)
    assert "cross_symbol_features" in bull
    assert len(extract_meta_feature_vector(bull)) == META_FEATURE_DIM
    args = client.predict_meta_batch.await_args[0][0]
    assert len(args[0][0]["feature_vector"]) == META_FEATURE_DIM


def test_apply_meta_payoff_to_metrics_put_side():
    metrics = {}
    score = apply_meta_payoff_to_metrics(
        metrics,
        direction=TradeDirection.PUT,
        tcn_probability=0.35,
        payoff_score=0.72,
        meta_applied=True,
    )
    assert score == pytest.approx(0.72)
    assert metrics["direction_put_score"] == pytest.approx(0.72)


def test_resolve_meta_payoff_score_uses_prefetched_value():
    metrics = {"meta_calibrated_payoff_score": 0.68, "meta_classifier_applied": True}
    score, applied = resolve_meta_payoff_score(
        symbol="RDBULL",
        metrics=metrics,
        direction=TradeDirection.CALL,
        tcn_probability=0.62,
        base_score=0.62,
        config={"infra": {"meta_classifier": {"enabled": True}}},
    )
    assert score == pytest.approx(0.68)
    assert applied is True


def test_resolve_meta_payoff_score_fetches_when_enabled():
    metrics = _metrics_with_cross()
    cfg = {"infra": {"meta_classifier": {"enabled": True, "http_url": "http://localhost:8005"}}}
    with patch(
        "src.application.services.meta_classifier_stacking.predict_meta_via_config_sync",
        return_value={"calibrated_payoff_score": 0.74, "meta_applied": True},
    ):
        score, applied = resolve_meta_payoff_score(
            symbol="RDBULL",
            metrics=metrics,
            direction=TradeDirection.CALL,
            tcn_probability=0.62,
            base_score=0.62,
            config=cfg,
        )
    assert score == pytest.approx(0.74)
    assert applied is True


def test_resolve_meta_payoff_score_exception_returns_fallback():
    metrics = _metrics_with_cross()
    cfg = {"infra": {"meta_classifier": {"enabled": True}}}
    with patch(
        "src.application.services.meta_classifier_stacking.predict_meta_via_config_sync",
        side_effect=ValueError("boom"),
    ):
        score, applied = resolve_meta_payoff_score(
            symbol="RDBULL",
            metrics=metrics,
            direction=TradeDirection.CALL,
            tcn_probability=0.62,
            base_score=0.62,
            config=cfg,
        )
    assert applied is False
    assert score == pytest.approx(0.62)


@pytest.mark.asyncio
async def test_prefetch_skips_invalid_entries_and_empty_batch():
    cfg = {"infra": {"meta_classifier": {"enabled": True}}}
    decisions = {
        "BAD": "not-a-dict",
        "NOMET": {"direction": TradeDirection.CALL},
        "NODIR": {"metrics": {"calibrated_prob": 0.6}},
        "NOPROB": {"direction": TradeDirection.CALL, "metrics": {}},
    }
    with patch(
        "src.application.services.meta_classifier_stacking.get_meta_classifier_client",
        new_callable=AsyncMock,
    ) as get_client:
        client = MagicMock()
        client.predict_meta_batch = AsyncMock()
        get_client.return_value = client
        await prefetch_meta_payoff_for_decisions(decisions, cfg)
    client.predict_meta_batch.assert_not_called()


@pytest.mark.asyncio
async def test_prefetch_disabled_returns_early():
    cfg = {"infra": {"meta_classifier": {"enabled": False}}}
    with patch(
        "src.application.services.meta_classifier_stacking.get_meta_classifier_client",
        new_callable=AsyncMock,
    ) as get_client:
        await prefetch_meta_payoff_for_decisions({}, cfg)
    get_client.assert_not_called()
