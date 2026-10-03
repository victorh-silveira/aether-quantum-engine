"""Testes unitarios para o classificador e gate de regime de mercado regime_ensemble."""

from src.domain.analytics.regime_ensemble import (
    REGIME_BALANCED,
    REGIME_COMPRESSION_RISK,
    REGIME_EXPLOSION_MOMENTUM,
    apply_regime_ensemble_gate,
    classify_market_regime,
)


def test_classify_market_regime_defaults_to_balanced():
    res = classify_market_regime(None)
    assert res["regime"] == REGIME_BALANCED
    assert res["doji_risk"] is False
    assert res["momentum_boost"] is False
    assert res["recommendation"] == "standard"


def test_classify_market_regime_compression():
    res = classify_market_regime(0.35)
    assert res["regime"] == REGIME_COMPRESSION_RISK
    assert res["doji_risk"] is True
    assert res["recommendation"] == "veto_breakout"

    res_hurst = classify_market_regime(0.80, hurst=0.40, bb_width=0.02)
    assert res_hurst["regime"] == REGIME_COMPRESSION_RISK
    assert res_hurst["doji_risk"] is True


def test_classify_market_regime_explosion():
    res = classify_market_regime(1.75, adx=0.30)
    assert res["regime"] == REGIME_EXPLOSION_MOMENTUM
    assert res["doji_risk"] is False
    assert res["momentum_boost"] is True
    assert res["recommendation"] == "prioritize_trend"


def test_apply_regime_ensemble_gate_veto_on_compression():
    metrics = {}
    vetoed, reason = apply_regime_ensemble_gate(metrics, 0.30, veto_on_compression=True)
    assert vetoed is True
    assert reason == "regime_compression_doji_risk"
    assert metrics["regime_ensemble"] == REGIME_COMPRESSION_RISK
    assert metrics["regime_ensemble_doji_risk"] is True

    vetoed_disabled, _ = apply_regime_ensemble_gate({}, 0.30, veto_on_compression=False)
    assert vetoed_disabled is False


def test_apply_regime_ensemble_gate_allows_explosion():
    metrics = {}
    vetoed, reason = apply_regime_ensemble_gate(metrics, 1.80, adx=0.35)
    assert vetoed is False
    assert reason == "ok"
    assert metrics["regime_ensemble"] == REGIME_EXPLOSION_MOMENTUM
    assert metrics["regime_ensemble_momentum_boost"] is True
