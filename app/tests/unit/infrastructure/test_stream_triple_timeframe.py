"""Testes de granularidade tripla e roteamento OHLC."""

from src.infrastructure.handlers.stream_timeframe import (
    ohlc_payload_granularity,
    resolve_mini_fetch_count,
    resolve_mini_granularity,
    resolve_triple_granularity,
)


def test_resolve_triple_and_mini():
    cfg = {"granularity": 600, "micro_granularity": 120, "mini_granularity": 60}
    macro, micro, mini = resolve_triple_granularity(cfg)
    assert (macro, micro, mini) == (600, 120, 60)
    assert resolve_mini_granularity({}) == 60
    assert resolve_mini_fetch_count({"mini_fetch_count": 100}) == 100


def test_ohlc_payload_prefers_explicit_granularity():
    assert ohlc_payload_granularity({"granularity": 60, "open_time": 120}, 600, 120, 60) == 60
    assert ohlc_payload_granularity({"open_time": 600}, 600, 120, 60) == 600
    assert ohlc_payload_granularity({"open_time": 120}, 600, 120, 60) == 120
    assert ohlc_payload_granularity({"open_time": 61}, 600, 120, 60) == 60
