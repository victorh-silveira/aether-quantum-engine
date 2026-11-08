"""Testes para o adaptador de mercado real e fluxos residuais do indicador."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from test_pullback import call_bars

from indicator.domain import Candle, synthetic_symbols
from indicator.pullback import MIN_BARS, evaluate
from indicator.pullback_training import diagnose_all
from indicator.real_market_feed import fetch_real_candles, resolve_ticker
from indicator.training import history


def test_resolve_ticker():
    assert resolve_ticker("OTC_SPC") == "^GSPC"
    assert resolve_ticker("UNKNOWN") is None


@pytest.mark.asyncio
async def test_fetch_real_candles_success(monkeypatch):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "chart": {
            "result": [
                {
                    "timestamp": [1700000000, 1700000900],
                    "indicators": {
                        "quote": [
                            {
                                "open": [100.0, 101.0],
                                "high": [102.0, 103.0],
                                "low": [99.0, 100.0],
                                "close": [101.0, 102.0],
                            }
                        ]
                    },
                }
            ]
        }
    }
    client_mock = AsyncMock()
    client_mock.get.return_value = mock_resp
    client_mock.__aenter__.return_value = client_mock
    client_mock.__aexit__.return_value = None
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: client_mock)

    candles = await fetch_real_candles("OTC_SPC", period_seconds=900)
    assert len(candles) == 2
    assert candles[0].close == 101.0


@pytest.mark.asyncio
async def test_fetch_real_candles_edges_and_errors(monkeypatch):
    assert await fetch_real_candles("UNKNOWN") == []

    mock_resp = MagicMock()
    mock_resp.status_code = 500
    client_mock = AsyncMock()
    client_mock.get.return_value = mock_resp
    client_mock.__aenter__.return_value = client_mock
    client_mock.__aexit__.return_value = None
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: client_mock)
    assert await fetch_real_candles("OTC_SPC") == []

    mock_resp.status_code = 200
    mock_resp.json.return_value = {"chart": {"result": []}}
    assert await fetch_real_candles("OTC_SPC") == []

    mock_resp.json.return_value = {
        "chart": {
            "result": [
                {
                    "timestamp": [1700000000, 1700000900, 1700001800],
                    "indicators": {
                        "quote": [
                            {
                                "open": [100.0, None],
                                "high": [102.0, 103.0],
                                "low": [99.0, 100.0],
                                "close": [101.0, 102.0],
                            }
                        ]
                    },
                }
            ]
        }
    }
    candles = await fetch_real_candles("OTC_SPC")
    assert len(candles) == 1

    client_mock.get.side_effect = RuntimeError("network_error")
    assert await fetch_real_candles("OTC_SPC") == []


@pytest.mark.asyncio
async def test_training_history_keeps_deriv_ohlc_isolated(monkeypatch):
    source = MagicMock()
    epoch = 900 * 2000
    source.candles = AsyncMock(return_value=[{"epoch": epoch, "open": 100, "high": 101, "low": 99, "close": 100}])
    candles = await history(source, "OTC_SPC", now=epoch + 1800, pages=1, period=900)
    assert len(candles) == 1
    assert candles[0].close == 100
    source.candles.assert_awaited_once()


def test_domain_scopes():
    payload = [
        {"symbol": "OTC_SPC", "market": "indices", "exchange_is_open": 1, "submarket": "americas_indices"},
        {"symbol": "R_100", "market": "synthetic_index", "exchange_is_open": 1},
    ]
    assert "OTC_SPC" in synthetic_symbols(payload, scope="stock_indices")
    assert "R_100" in synthetic_symbols(payload, scope="volatility_indices")


def test_pullback_organic_scoring_and_gaps():
    bars = call_bars(300)
    now = MIN_BARS * 300
    res_call = evaluate(bars, 300, now, "improved")
    assert res_call.side in ("CALL", "PUT")

    mirrored = [Candle(bar.epoch, 200 - bar.open, 200 - bar.low, 200 - bar.high, 200 - bar.close) for bar in bars]
    res_put = evaluate(mirrored, 300, now, "improved")
    assert res_put.side in ("CALL", "PUT")

    bars_gap = list(bars)
    bars_gap[-1] = Candle(bars[-2].epoch + 600, bars[-1].open, bars[-1].high, bars[-1].low, bars[-1].close)
    res_gap = evaluate(bars_gap, 300, bars_gap[-1].epoch + 300, "improved")
    assert res_gap.side == "SEM SINAL"
    assert res_gap.reason == "lacuna"


def test_pullback_improved_tie_breakers():
    bars = call_bars(300)
    now = MIN_BARS * 300

    with (
        patch("indicator.pullback._rsi", return_value=10.0),
        patch("indicator.pullback._adx", return_value=(5.0, 5.0, 5.0)),
        patch("indicator.pullback._stochastic", return_value=(5.0, 5.0)),
    ):
        res = evaluate(bars, 300, now, "improved")
        assert res.side in ("CALL", "PUT")

    last = bars[-1]
    c_down = Candle(last.epoch, last.open, last.open + 0.1, last.open - 2.0, last.open - 1.0)
    bars_down = list(bars[:-1]) + [c_down]
    with (
        patch("indicator.pullback._rsi", return_value=10.0),
        patch("indicator.pullback._adx", return_value=(5.0, 5.0, 5.0)),
        patch("indicator.pullback._stochastic", return_value=(5.0, 5.0)),
        patch("indicator.pullback._ema", return_value=[last.open + 10.0] * MIN_BARS),
    ):
        res_down = evaluate(bars_down, 300, now, "improved")
        assert res_down.side in ("CALL", "PUT")

    c_above = Candle(last.epoch, 104.0, 106.0, 103.0, 105.0)
    bars_above = list(bars[:-1]) + [c_above]
    with (
        patch("indicator.pullback._rsi", return_value=0.0),
        patch("indicator.pullback._cmo", return_value=-50.0),
        patch("indicator.pullback._williams_r", return_value=0.0),
        patch("indicator.pullback._bollinger_pct_b", return_value=0.0),
        patch("indicator.pullback._adx", return_value=(0.0, 0.0, 0.0)),
        patch("indicator.pullback._stochastic", return_value=(0.0, 0.0)),
        patch("indicator.pullback._ema", return_value=[100.0] * MIN_BARS),
    ):
        res_above = evaluate(bars_above, 300, now, "improved")
        assert res_above.side == "CALL"

    c_below = Candle(last.epoch, 96.0, 97.0, 94.0, 95.0)
    bars_below = list(bars[:-1]) + [c_below]
    with (
        patch("indicator.pullback._rsi", return_value=0.0),
        patch("indicator.pullback._cmo", return_value=-50.0),
        patch("indicator.pullback._williams_r", return_value=0.0),
        patch("indicator.pullback._bollinger_pct_b", return_value=0.0),
        patch("indicator.pullback._adx", return_value=(0.0, 0.0, 0.0)),
        patch("indicator.pullback._stochastic", return_value=(0.0, 0.0)),
        patch("indicator.pullback._ema", return_value=[100.0] * MIN_BARS),
    ):
        res_below = evaluate(bars_below, 300, now, "improved")
        assert res_below.side == "PUT"

    c_flat1 = Candle(last.epoch, 100.0, 101.0, 99.0, 100.0)

    bars_flat1 = list(bars[:-1]) + [c_flat1]
    with (
        patch("indicator.pullback._rsi", return_value=10.0),
        patch("indicator.pullback._adx", return_value=(5.0, 5.0, 5.0)),
        patch("indicator.pullback._stochastic", return_value=(5.0, 5.0)),
        patch("indicator.pullback._ema", return_value=[100.0] * MIN_BARS),
    ):
        res_flat1 = evaluate(bars_flat1, 300, now, "improved")
        assert res_flat1.side in ("CALL", "PUT")

    c_flat2 = Candle(last.epoch, 100.5, 101.0, 99.0, 100.0)
    bars_flat2 = list(bars[:-1]) + [c_flat2]
    with (
        patch("indicator.pullback._rsi", return_value=10.0),
        patch("indicator.pullback._adx", return_value=(5.0, 5.0, 5.0)),
        patch("indicator.pullback._stochastic", return_value=(5.0, 5.0)),
        patch("indicator.pullback._ema", return_value=[100.0] * MIN_BARS),
    ):
        res_flat2 = evaluate(bars_flat2, 300, now, "improved")
        assert res_flat2.side in ("CALL", "PUT")


@pytest.mark.asyncio
async def test_diagnose_stock_gap():
    period = 900
    candles = [
        {
            "epoch": 1700000000 + i * period if i != 130 else 1700000000 + i * period + 3600,
            "open": 100.0 + i,
            "high": 101.0 + i,
            "low": 99.0 + i,
            "close": 100.5 + i,
        }
        for i in range(250)
    ]
    source = MagicMock()
    source.symbols = AsyncMock(
        return_value=[{"market": "indices", "underlying_symbol": "OTC_SPC", "exchange_is_open": 1}]
    )
    source.candles = AsyncMock(return_value=candles)

    store = MagicMock()
    store.set_symbols = AsyncMock()
    store.register_setups = AsyncMock()
    store.save_candles = AsyncMock()

    with patch("indicator.pullback_training.history", AsyncMock(return_value=[Candle(**c) for c in candles])):
        report = await diagnose_all(source, store, scope="stock_indices", now=1700000000 + 260 * period)
        assert report is not None
