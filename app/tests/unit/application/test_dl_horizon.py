from src.application.services.deep_learning.dl_horizon import (
    contract_duration_seconds,
    resolve_label_horizon_bars,
)


def test_contract_duration_seconds_minutes():
    assert contract_duration_seconds({"duration": 5, "duration_unit": "m"}) == 300


def test_label_horizon_bars_aligns_contract_to_granularity():
    risk = {"duration": 1, "duration_unit": "m"}
    assert resolve_label_horizon_bars(60, risk, {}) == 1
    assert resolve_label_horizon_bars(300, risk, {}) == 1
    assert resolve_label_horizon_bars(60, {"duration": 5, "duration_unit": "m"}, {}) == 5


def test_label_horizon_explicit_override():
    assert resolve_label_horizon_bars(60, {"duration": 1, "duration_unit": "m"}, {"label_horizon_bars": 3}) == 3


def test_contract_duration_seconds_and_hours():
    assert contract_duration_seconds({"duration": 45, "duration_unit": "seconds"}) == 45
    assert contract_duration_seconds({"duration": 2, "duration_unit": "hour"}) == 7200
