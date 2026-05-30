import pytest

from scripts.backtest.gemini_collect import estimate_gemini_minutes
from scripts.backtest.gemini_schedule import gemini_query_points, payload_for_bar
from scripts.backtest.timeframe import bars_per_day, primary_granularity_seconds


def test_gemini_query_points_daily():
    per_day = bars_per_day(primary_granularity_seconds({}))
    points = gemini_query_points(10, 10 + per_day * 2, "daily", 1)
    assert points == [10, 10 + per_day, 10 + per_day * 2]


def test_payload_for_bar_daily_reuses_day_decision():
    per_day = bars_per_day(primary_granularity_seconds({}))
    start = 100
    day2 = start + per_day
    cache = {
        str(day2): {
            "_direction_normalized": "CALL",
            "_conviction_normalized": 0.8,
            "us_cluster": "CALL",
            "eu_cluster": "CALL",
        }
    }
    payload = payload_for_bar(cache, day2 + 10, start, "daily", 1, [day2])
    assert payload is not None
    assert payload["_direction_normalized"] == "CALL"


def test_estimate_gemini_minutes_pending_only():
    minutes = estimate_gemini_minutes(10, cached=7, seconds_per_call=60.0)
    assert minutes == pytest.approx(3.0)
