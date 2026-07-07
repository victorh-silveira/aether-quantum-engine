from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.deep_learning.dl_predict_build import prepare_meta_classifier_cross_symbol_bundle
from src.application.services.execution_direction_resolver import resolve_execution_direction
from src.application.services.meta_classifier_cross_symbol import attach_cross_symbol_features_to_decisions
from src.application.services.meta_classifier_features import (
    META_FEATURE_DIM,
    cross_symbol_conviction_spread,
    extract_meta_feature_vector,
    side_payoff_from_probability,
)
from src.application.services.meta_classifier_stacking import (
    apply_meta_regression_edge_to_metrics,
    prefetch_meta_payoff_for_decisions,
    resolve_meta_payoff_edge,
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


def test_cross_symbol_conviction_spread_reads_attached_triplet():
    metrics = _metrics_with_cross()
    assert cross_symbol_conviction_spread(metrics) == pytest.approx(0.21)


def test_cross_symbol_conviction_spread_defaults_without_triplet():
    assert cross_symbol_conviction_spread({}) == 0.0


def test_parallel_drift_regime_exposes_low_relative_conviction_spread():
    decisions = {
        "RDBULL": {
            "direction": TradeDirection.CALL,
            "metrics": {
                "calibrated_prob": 0.62,
                "micro_indicators": {"rsi": 58.0, "vol_ratio": 1.05},
                "feature_vector": [0.1] * 34,
            },
        },
        "RDBEAR": {
            "direction": TradeDirection.CALL,
            "metrics": {
                "calibrated_prob": 0.39,
                "micro_indicators": {"rsi": 44.0, "vol_ratio": 0.92},
                "feature_vector": [0.2] * 34,
            },
        },
    }
    attach_cross_symbol_features_to_decisions(decisions)
    spread = cross_symbol_conviction_spread(decisions["RDBULL"]["metrics"])
    assert spread == pytest.approx(abs(0.62 - (1.0 - 0.39)))
    assert spread < 0.05


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
                {"predicted_payoff_edge": 0.12, "meta_applied": True},
                {"predicted_payoff_edge": 0.08, "meta_applied": True},
            ]
        )
        get_client.return_value = client
        await prefetch_meta_payoff_for_decisions(decisions, cfg)
    bull = decisions["RDBULL"]["metrics"]
    bear = decisions["RDBEAR"]["metrics"]
    assert bull["predicted_payoff_edge"] == pytest.approx(0.12)
    assert bull["trade_score"] == pytest.approx(0.62)
    assert bear["predicted_payoff_edge"] == pytest.approx(0.08)
    assert bear["trade_score"] == pytest.approx(0.61)
    assert "cross_symbol_features" in bull
    assert len(extract_meta_feature_vector(bull)) == META_FEATURE_DIM
    args = client.predict_meta_batch.await_args[0][0]
    assert len(args[0][0]["feature_vector"]) == META_FEATURE_DIM


@pytest.mark.asyncio
async def test_prefetch_meta_payoff_attaches_cross_symbol_when_missing():
    decisions = {
        "RDBULL": {
            "direction": TradeDirection.CALL,
            "metrics": {
                "calibrated_prob": 0.66,
                "feature_vector": [0.1] * 34,
                "micro_indicators": {"rsi": 60.0, "vol_ratio": 1.1},
            },
        },
        "RDBEAR": {
            "direction": TradeDirection.PUT,
            "metrics": {
                "calibrated_prob": 0.38,
                "feature_vector": [0.2] * 34,
                "micro_indicators": {"rsi": 42.0, "vol_ratio": 0.9},
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
                {"predicted_payoff_edge": 0.11, "meta_applied": True},
                {"predicted_payoff_edge": 0.05, "meta_applied": True},
            ]
        )
        get_client.return_value = client
        prepare_meta_classifier_cross_symbol_bundle(MagicMock(), decisions, {"micro_granularity": 60})
        await prefetch_meta_payoff_for_decisions(decisions, cfg)
    assert "cross_symbol_features" in decisions["RDBULL"]["metrics"]


def test_prepare_meta_classifier_bundle_skips_invalid_entries():
    decisions = {"BAD": "x", "EMPTY": {"metrics": "invalid"}}
    prepare_meta_classifier_cross_symbol_bundle(MagicMock(), decisions, {"micro_granularity": 60})
    assert decisions["BAD"] == "x"


def test_apply_meta_regression_edge_to_metrics_put_side():
    metrics = {}
    score = apply_meta_regression_edge_to_metrics(
        metrics,
        direction=TradeDirection.PUT,
        tcn_probability=0.35,
        predicted_edge=0.14,
        meta_applied=True,
        base_score=0.65,
    )
    assert score == pytest.approx(0.65)
    assert metrics["direction_put_score"] == pytest.approx(0.65)


def test_resolve_meta_payoff_edge_uses_prefetched_value():
    metrics = {"predicted_payoff_edge": 0.18, "meta_classifier_applied": True}
    edge, applied = resolve_meta_payoff_edge(
        symbol="RDBULL",
        metrics=metrics,
        direction=TradeDirection.CALL,
        tcn_probability=0.62,
        _base_score=0.62,
        config={"infra": {"meta_classifier": {"enabled": True}}},
    )
    assert edge == pytest.approx(0.18)
    assert applied is True


def test_resolve_meta_payoff_edge_without_prefetch_returns_neutral():
    metrics = _metrics_with_cross()
    cfg = {"infra": {"meta_classifier": {"enabled": True, "http_url": "http://localhost:8005"}}}
    edge, applied = resolve_meta_payoff_edge(
        symbol="RDBULL",
        metrics=metrics,
        direction=TradeDirection.CALL,
        tcn_probability=0.62,
        _base_score=0.62,
        config=cfg,
    )
    assert edge == pytest.approx(0.0)
    assert applied is False


def test_resolve_meta_payoff_edge_without_prefetch_even_when_enabled():
    metrics = _metrics_with_cross()
    cfg = {"infra": {"meta_classifier": {"enabled": True}}}
    edge, applied = resolve_meta_payoff_edge(
        symbol="RDBULL",
        metrics=metrics,
        direction=TradeDirection.CALL,
        tcn_probability=0.62,
        _base_score=0.62,
        config=cfg,
    )
    assert applied is False
    assert edge == pytest.approx(0.0)


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


def test_c0015_stacking_payload_rejects_negative_edge_before_squeeze(caplog):
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "calibrated_prob": 0.70,
            "predicted_payoff_edge": -0.22,
            "meta_classifier_applied": True,
            "feature_vector": [0.1] * 34,
            "indicators": {"bb_width": 0.03},
            "flow_features": {"micro_tick_acceleration": -0.02, "keltner_deviation_ratio": -0.05},
            "cross_symbol_features": {
                "cross_symbol_prob_delta": 0.12,
                "cross_symbol_vol_ratio_diff": -0.04,
                "cross_symbol_rsi_spread": 3.0,
            },
        },
    }
    with caplog.at_level("INFO"):
        result = resolve_execution_direction(entry, symbol="RDBULL")
    assert result is None
    assert entry["metrics"]["quality_guard_reject"] is True
    assert entry["metrics"]["signal_status"] == "SIGNAL_SUSPENDED"
    assert len(extract_meta_feature_vector(entry["metrics"])) == META_FEATURE_DIM
    assert not any("[D-SQUEEZE]" in record.message for record in caplog.records)
