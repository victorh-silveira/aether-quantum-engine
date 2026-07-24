import numpy as np
import pytest

from src.application.services.deep_learning import dl_trend as dl_trend_mod
from src.application.services.deep_learning.dl_indicator_config import load_indicator_config_from_settings
from src.application.services.deep_learning.dl_trend import calculate_trend_direction, consensus_trend_direction
from src.domain.models.trade import TradeDirection


def test_consensus_trend_direction_call_vote_increment():
    consensus = load_indicator_config_from_settings()["trend_consensus"]
    series = {
        "di_diff": [0.1],
        "macd": [0.3],
        "macd_signal": [0.2],
        "rsi": [0.6],
        "cmo": [0.1],
        "keltner_pct_b": [0.6],
    }
    direction, call_votes, _ = consensus_trend_direction(TradeDirection.PUT, series, consensus)
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
    direction, call_votes, put_votes = consensus_trend_direction(TradeDirection.CALL, series, consensus)
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
    direction, _, _, _, _ = calculate_trend_direction(prices, {}, {"trend_period": 15, "trend_use_slope": False})
    assert direction == TradeDirection.CALL


def test_calculate_trend_direction_ema_single_bar_and_sma_empty():
    assert dl_trend_mod._ema_tail(np.array([10.0]), 5) == 10.0
    assert dl_trend_mod._sma_tail(np.array([], dtype=np.float64), 5) == 0.0


def test_calculate_trend_direction_requires_trend_period(monkeypatch):
    monkeypatch.setattr(dl_trend_mod, "_load_execution_trend_defaults", lambda: {})
    with pytest.raises(KeyError, match="trend_period"):
        calculate_trend_direction(np.array([1.0, 2.0]), {}, {})


def test_safe_last_empty_or_none():
    assert dl_trend_mod._safe_last({}, "di_diff") is None
    assert dl_trend_mod._safe_last({"di_diff": []}, "di_diff") is None
    assert dl_trend_mod._safe_last({"di_diff": [0.05]}, "di_diff") == 0.05
