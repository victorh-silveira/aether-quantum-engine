import numpy as np

from src.application.services.deep_learning.dl_trend import calculate_trend_direction, consensus_trend_direction
from src.domain.models.trade import TradeDirection


def test_consensus_trend_direction_call_vote_increment():
    series = {
        "di_diff": [0.1],
        "macd": [0.3],
        "macd_signal": [0.2],
        "rsi": [0.6],
        "cmo": [0.1],
        "keltner_pct_b": [0.6],
    }
    direction, call_votes, _ = consensus_trend_direction(TradeDirection.PUT, series)
    assert direction == TradeDirection.CALL
    assert call_votes > 1
    series = {
        "di_diff": [-0.1],
        "macd": [0.1],
        "macd_signal": [0.2],
        "rsi": [0.4],
        "cmo": [-0.1],
        "keltner_pct_b": [0.4],
    }
    direction, call_votes, put_votes = consensus_trend_direction(TradeDirection.CALL, series)
    assert direction == TradeDirection.PUT
    assert put_votes > call_votes


def test_calculate_trend_direction_sma_without_slope():
    prices = np.array([10.0, 9.0, 8.0, 7.0])
    series = {}
    direction, trend_type, period, _, _ = calculate_trend_direction(
        prices,
        series,
        {"trend_period": 2, "trend_use_ema": False, "trend_use_slope": False},
    )
    assert direction == TradeDirection.PUT
    assert trend_type == "SMA"


def test_calculate_trend_direction_empty_prices_branch():
    prices = np.array([], dtype=np.float64)
    direction, _, _, _, _ = calculate_trend_direction(prices, {}, {"trend_use_slope": False})
    assert direction == TradeDirection.CALL
