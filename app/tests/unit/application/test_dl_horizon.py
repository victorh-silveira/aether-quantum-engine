from src.application.services.deep_learning.dl_horizon import (
    contract_duration_seconds,
    resolve_label_horizon_bars,
)


def test_contract_duration_seconds_minutes():
    assert contract_duration_seconds({"duration": 5, "duration_unit": "m"}) == 300


def test_label_horizon_bars_aligns_contract_to_granularity():
    risk = {"duration": 1, "duration_unit": "m"}
    assert resolve_label_horizon_bars(60, risk, {}) == 1
    assert resolve_label_horizon_bars(900, risk, {}) == 1
    assert resolve_label_horizon_bars(60, {"duration": 5, "duration_unit": "m"}, {}) == 5


def test_label_horizon_explicit_override():
    assert resolve_label_horizon_bars(60, {"duration": 1, "duration_unit": "m"}, {"label_horizon_bars": 3}) == 3


def test_contract_duration_seconds_and_hours():
    assert contract_duration_seconds({"duration": 45, "duration_unit": "seconds"}) == 45
    assert contract_duration_seconds({"duration": 2, "duration_unit": "hour"}) == 7200


def test_contract_duration_seconds_ticks():
    assert contract_duration_seconds({"duration": 1, "duration_unit": "t"}) == 2
    assert contract_duration_seconds({"duration": 10, "duration_unit": "t"}) == 20
    assert contract_duration_seconds({"duration": 1, "duration_unit": "d"}) == 86400


def test_label_horizon_bars_for_tick_contract():
    risk = {"duration": 1, "duration_unit": "t"}
    assert resolve_label_horizon_bars(60, risk, {}) == 1
    assert resolve_label_horizon_bars(10, risk, {}) == 1


def test_hybrid_contract_30s_micro_60s_horizon_rounds_to_one_bar():
    risk = {"duration": 30, "duration_unit": "s"}
    assert contract_duration_seconds(risk) == 30
    assert resolve_label_horizon_bars(60, risk, {}) == 1
    assert resolve_label_horizon_bars(60, risk, {"label_horizon_bars": 1}) == 1
