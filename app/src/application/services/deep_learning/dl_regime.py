"""Filtros de regime e momentum para confirmar direcao do modelo."""

import numpy as np

from src.application.services.deep_learning.dl_features import precompute_price_series
from src.domain.models.trade import TradeDirection


def latest_momentum(prices: np.ndarray) -> tuple[float, float]:
    """Retorna ema_spread e retorno de 5 barras na ultima vela."""
    if len(prices) < 6:
        return 0.0, 0.0
    series = precompute_price_series(prices)
    idx = len(prices) - 1
    return float(series["ema_spread"][idx]), float(series["ret_5"][idx])


def direction_aligns_with_regime(
    direction: TradeDirection,
    prices: np.ndarray,
    *,
    min_strength: float = 0.0,
) -> bool:
    """Exige momentum de curto prazo coerente com CALL ou PUT."""
    ema_spread, ret_5 = latest_momentum(prices)
    strength = max(0.0, float(min_strength))
    if direction == TradeDirection.CALL:
        return ema_spread >= strength and ret_5 >= strength
    return ema_spread <= -strength and ret_5 <= -strength


def regime_strength(prices: np.ndarray) -> float:
    """Magnitude combinada do sinal de regime na ultima barra."""
    ema_spread, ret_5 = latest_momentum(prices)
    return max(abs(ema_spread), abs(ret_5))
