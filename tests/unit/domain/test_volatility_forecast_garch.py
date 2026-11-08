"""Testes unitarios para o previsor de volatilidade realizada."""

import numpy as np

from src.domain.analytics.volatility_forecast_garch import HighFrequencyVolatilityForecaster


def test_volatility_forecast_short_prices():
    forecaster = HighFrequencyVolatilityForecaster()
    forecast = forecaster.forecast_next_bar_rv([100.0, 100.1])

    assert forecast.predicted_rv_300s > 0.0
    assert not forecast.is_compression_risk
    assert forecast.confidence_score == 0.50


def test_volatility_forecast_compression_detection():
    constant_prices = [100.0] * 100
    forecaster = HighFrequencyVolatilityForecaster(compression_threshold=0.002)
    forecast = forecaster.forecast_next_bar_rv(constant_prices)

    assert forecast.current_rv_60s == 0.0
    assert forecast.is_compression_risk


def test_volatility_forecast_dynamic_scaling():
    rng = np.random.default_rng(123)
    noisy_prices = 100.0 + np.cumsum(rng.normal(0, 0.5, 350))
    forecaster = HighFrequencyVolatilityForecaster()
    forecast = forecaster.forecast_next_bar_rv(noisy_prices)

    assert forecast.predicted_rv_300s > 0.0
    assert forecast.confidence_score == 0.85
    assert not forecast.is_compression_risk
