from scripts.backtest.timeframe import bars_per_day, primary_granularity_seconds


def test_bars_per_day_m5():
    cfg = {"data_handler": {"granularity": 300}}
    assert primary_granularity_seconds(cfg) == 300
    assert bars_per_day(300) == 288


def test_bars_per_day_m15_default():
    assert bars_per_day(primary_granularity_seconds({})) == 96
