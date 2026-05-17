from datetime import datetime

from src.domain.models.market_data import Candle, MarketSeries, Tick


def test_market_series_records_ticks_and_candles():
    ms = MarketSeries(symbol="TEST")
    t = Tick(symbol="TEST", quote=1.0, time=datetime.now(), epoch=1000)
    ms.ticks.append(t)

    rec_ticks = ms.records("ticks")
    assert len(rec_ticks) == 1
    assert rec_ticks[0]["quote"] == 1.0

    c = Candle(symbol="TEST", open=1.0, high=1.2, low=0.9, close=1.1, time=datetime.now(), epoch=1000)
    ms.candles.append(c)

    rec_c = ms.records("candles")
    assert len(rec_c) == 1
    assert rec_c[0]["close"] == 1.1
