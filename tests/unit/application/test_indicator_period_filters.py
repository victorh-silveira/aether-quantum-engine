"""Testes unitarios para os filtros e formatadores de ordens do indicador."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from src.application.services.orchestrator.indicator_period_filters import (
    build_indicator_metrics,
    closed_candles_slice,
    is_promoted_signal_viable,
)


def test_closed_candles_slice_short_list():
    candles = [SimpleNamespace(epoch=100) for _ in range(10)]
    assert closed_candles_slice(candles, 900, 1000.0) == candles
    assert closed_candles_slice([], 900, 1000.0) == []


def test_closed_candles_slice_discards_active_candle():
    period = 900
    now_ts = 180000.0
    current_block = int(now_ts // period) * period
    candles = [SimpleNamespace(epoch=100000 + i * 900) for i in range(55)]
    candles[-1].epoch = current_block
    res = closed_candles_slice(candles, period, now_ts)
    assert len(res) == 54


def test_closed_candles_slice_keeps_closed_candle():
    period = 900
    now_ts = 180000.0
    current_block = int(now_ts // period) * period
    candles = [SimpleNamespace(epoch=100000 + i * 900) for i in range(55)]
    candles[-1].epoch = current_block - 900
    res = closed_candles_slice(candles, period, now_ts)
    assert len(res) == 55


def test_build_indicator_metrics_call_and_put():
    ev = SimpleNamespace(settled=10, lower_95=0.05, exploratory_purchased=1)
    metrics_call = build_indicator_metrics(
        1700000000, 900, "v1", "CALL", "CALL", "motivo", pilot=True, ev=ev, prob=0.65
    )
    assert metrics_call["indicator_controls_execution"] is True
    assert metrics_call["indicator_execution_side"] == "CALL"
    assert metrics_call["calibrated_prob"] == 0.65
    assert metrics_call["duration"] == 15

    metrics_put = build_indicator_metrics(1700000000, 900, "v1", "PUT", "PUT", "motivo", pilot=False, ev=ev, prob=0.65)
    assert metrics_put["indicator_execution_side"] == "PUT"
    assert metrics_put["calibrated_prob"] == 0.35


def test_is_promoted_signal_viable_none_or_closed(monkeypatch):
    orch = SimpleNamespace(config={"pullback_strategy": {"allow_closed_markets": False}})
    assert is_promoted_signal_viable(orch, None, 900, 1700000000.0) is False

    monkeypatch.setattr(
        "src.application.services.orchestrator.indicator_period_filters.is_otc_exchange_open_brt",
        lambda sym, ts: False,
    )
    assert is_promoted_signal_viable(orch, "OTC_SSMI", 900, 1700000000.0) is False


def test_is_promoted_signal_viable_allow_closed_markets(monkeypatch):
    orch = SimpleNamespace(
        config={"pullback_strategy": {"allow_closed_markets": True, "version": "pullback-1-test"}},
        _indicator_period_signals={},
    )
    monkeypatch.setattr(
        "src.application.services.orchestrator.indicator_period_filters.is_otc_exchange_open_brt",
        lambda sym, ts: False,
    )
    assert is_promoted_signal_viable(orch, "OTC_SSMI", 900, 1700000000.0) is False

    monkeypatch.setattr(
        "src.application.services.orchestrator.indicator_period_filters.is_otc_exchange_open_brt",
        lambda sym, ts: True,
    )
    sig = SimpleNamespace(model_version="pullback-1-test", side="CALL", candle_epoch=1700000000)
    orch._indicator_period_signals = {("OTC_SSMI", 900): sig}
    fn = MagicMock(return_value=True)
    assert is_promoted_signal_viable(orch, "OTC_SSMI", 900, 1700000000.0, signal_epoch_fn=fn) is True

    sig_invalid = SimpleNamespace(model_version="other-v1", side="CALL", candle_epoch=1700000000)
    orch._indicator_period_signals = {("OTC_SSMI", 900): sig_invalid}
    assert is_promoted_signal_viable(orch, "OTC_SSMI", 900, 1700000000.0, signal_epoch_fn=fn) is False
