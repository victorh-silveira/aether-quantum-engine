"""Previsao de volatilidade realizada condicional baseada em HAR-RV e GARCH de microestrutura."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.domain.analytics.realized_volatility import compute_realized_volatility


@dataclass(frozen=True)
class VolatilityForecast:
    """Estimativa causal de volatilidade realizada para os proximos 300 segundos."""

    predicted_rv_300s: float
    current_rv_60s: float
    current_rv_300s: float
    is_compression_risk: bool
    confidence_score: float


class HighFrequencyVolatilityForecaster:
    """Modelo heterogeneo autorregressivo para previsao de volatilidade realizada (HAR-RV)."""

    def __init__(
        self,
        *,
        omega: float = 0.0005,
        alpha_fast: float = 0.65,
        beta_slow: float = 0.30,
        compression_threshold: float = 0.0015,
    ) -> None:
        """Inicializa pesos do estimador autoregressivo causal."""
        self._omega = float(omega)
        self._alpha = float(alpha_fast)
        self._beta = float(beta_slow)
        self._comp_thresh = float(compression_threshold)

    def forecast_next_bar_rv(
        self,
        recent_prices: list[float] | np.ndarray,
        *,
        fast_window: int = 60,
        slow_window: int = 300,
    ) -> VolatilityForecast:
        """Estima a volatilidade realizada da proxima barra M5 a partir da assinatura de micro-ticks."""
        prices = np.asarray(recent_prices, dtype=np.float64)
        if len(prices) < 5:
            return VolatilityForecast(
                predicted_rv_300s=0.0020,
                current_rv_60s=0.0020,
                current_rv_300s=0.0020,
                is_compression_risk=False,
                confidence_score=0.50,
            )

        fast_slice = prices[-fast_window:] if len(prices) >= fast_window else prices
        slow_slice = prices[-slow_window:] if len(prices) >= slow_window else prices

        rv_fast = compute_realized_volatility(fast_slice)
        rv_slow = compute_realized_volatility(slow_slice)

        predicted_rv = self._omega + (self._alpha * rv_fast) + (self._beta * rv_slow)
        predicted_rv = max(0.0001, float(predicted_rv))

        is_compression = predicted_rv < self._comp_thresh
        confidence = 0.85 if len(prices) >= slow_window else 0.60

        return VolatilityForecast(
            predicted_rv_300s=round(predicted_rv, 6),
            current_rv_60s=round(rv_fast, 6),
            current_rv_300s=round(rv_slow, 6),
            is_compression_risk=is_compression,
            confidence_score=confidence,
        )
