from src.domain.trading.elastic_regime import (
    MarketRegime,
    classify_market_regime,
    evaluate_elastic_direction,
)


def test_classify_market_regime_exhaustion():
    assert classify_market_regime(adx=30.0, spread=0.05, rsi=80.0, recuo=0.5) == MarketRegime.EXHAUSTION
    assert classify_market_regime(adx=30.0, spread=0.05, rsi=20.0, recuo=0.5) == MarketRegime.EXHAUSTION
    assert classify_market_regime(adx=30.0, spread=0.05, rsi=50.0, recuo=2.5) == MarketRegime.EXHAUSTION


def test_classify_market_regime_strong_trend():
    assert classify_market_regime(adx=25.0, spread=0.005, rsi=50.0, recuo=0.5) == MarketRegime.STRONG_TREND
    assert classify_market_regime(adx=15.0, spread=0.02, rsi=50.0, recuo=0.5) == MarketRegime.STRONG_TREND


def test_classify_market_regime_moderate_and_ranging():
    assert classify_market_regime(adx=20.0, spread=0.005, rsi=50.0, recuo=0.5) == MarketRegime.MODERATE_TREND
    assert classify_market_regime(adx=14.0, spread=0.005, rsi=50.0, recuo=0.5) == MarketRegime.RANGING


def test_classify_market_regime_defensive_types():
    assert classify_market_regime(adx=None, spread=None, rsi=None, recuo=None) == MarketRegime.MODERATE_TREND


def test_evaluate_elastic_direction_invalid_side():
    side, reason, flipped = evaluate_elastic_direction("INVALID", "ALTA", MarketRegime.STRONG_TREND)
    assert side == "INVALID"
    assert reason == "INVALID_SIDE"
    assert not flipped


def test_evaluate_elastic_direction_strong_trend_aligned():
    side, reason, flipped = evaluate_elastic_direction(
        "CALL", "ALTA", MarketRegime.STRONG_TREND, has_loss_reversal=True
    )
    assert side == "CALL"
    assert reason == "TREND_ALIGNED_DIRECT"
    assert not flipped

    side, reason, flipped = evaluate_elastic_direction(
        "PUT", "BAIXA", MarketRegime.STRONG_TREND, has_loss_reversal=False
    )
    assert side == "PUT"
    assert reason == "TREND_ALIGNED_DIRECT"
    assert not flipped


def test_evaluate_elastic_direction_strong_trend_unaligned():
    side, reason, flipped = evaluate_elastic_direction(
        "CALL", "BAIXA", MarketRegime.STRONG_TREND, has_loss_reversal=True
    )
    assert side == "PUT"
    assert reason == "REVERSAL_ON_LOSS"
    assert flipped

    side, reason, flipped = evaluate_elastic_direction(
        "CALL", "BAIXA", MarketRegime.STRONG_TREND, has_loss_reversal=False
    )
    assert side == "CALL"
    assert reason == "BASE_SIGNAL_DIRECT"
    assert not flipped


def test_evaluate_elastic_direction_exhaustion():
    side, reason, flipped = evaluate_elastic_direction("CALL", "ALTA", MarketRegime.EXHAUSTION)
    assert side == "PUT"
    assert reason == "ELASTIC_EXHAUSTION_INVERTED"
    assert flipped


def test_evaluate_elastic_direction_ranging():
    side, reason, flipped = evaluate_elastic_direction("CALL", "LATERAL", MarketRegime.RANGING, has_loss_reversal=False)
    assert side == "PUT"
    assert reason == "LATERAL_FADE_INVERTED"
    assert flipped

    side, reason, flipped = evaluate_elastic_direction("CALL", "LATERAL", MarketRegime.RANGING, has_loss_reversal=True)
    assert side == "PUT"
    assert reason == "REVERSAL_ON_LOSS"
    assert flipped


def test_evaluate_elastic_direction_moderate_trend():
    side, reason, flipped = evaluate_elastic_direction(
        "CALL", "ALTA", MarketRegime.MODERATE_TREND, has_loss_reversal=True
    )
    assert side == "PUT"
    assert reason == "REVERSAL_ON_LOSS"
    assert flipped

    side, reason, flipped = evaluate_elastic_direction(
        "CALL", "ALTA", MarketRegime.MODERATE_TREND, has_loss_reversal=False
    )
    assert side == "CALL"
    assert reason == "BASE_SIGNAL_DIRECT"
    assert not flipped
