"""Testes unitarios para o adaptador de mercado real (RealMarketFeed)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.infrastructure.adapters.real_market_feed import (
    RealMarketFeed,
    resolve_real_market_symbol,
)


def test_resolve_real_market_symbol():
    assert resolve_real_market_symbol("OTC_SPC") == "^GSPC"
    assert resolve_real_market_symbol("OTC_NDX") == "^NDX"
    assert resolve_real_market_symbol("OTC_DJI") == "^DJI"
    assert resolve_real_market_symbol("R_100") is None


@pytest.mark.asyncio
async def test_fetch_real_candles_success(monkeypatch):
    feed = RealMarketFeed()
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
                                "open": [5000.0, 5010.0],
                                "high": [5015.0, 5020.0],
                                "low": [4995.0, 5005.0],
                                "close": [5010.0, 5018.0],
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

    candles = await feed.fetch_real_candles("OTC_SPC", period_seconds=900)
    assert len(candles) == 2
    assert candles[0]["open"] == 5000.0
    assert candles[1]["close"] == 5018.0


@pytest.mark.asyncio
async def test_fetch_real_candles_unknown_symbol():
    feed = RealMarketFeed()
    candles = await feed.fetch_real_candles("UNKNOWN_ASSET")
    assert candles == []


@pytest.mark.asyncio
async def test_fetch_real_candles_http_error(monkeypatch):
    feed = RealMarketFeed()
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    client_mock = AsyncMock()
    client_mock.get.return_value = mock_resp
    client_mock.__aenter__.return_value = client_mock
    client_mock.__aexit__.return_value = None

    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: client_mock)

    candles = await feed.fetch_real_candles("OTC_SPC")
    assert candles == []


@pytest.mark.asyncio
async def test_fetch_real_quote_success(monkeypatch):
    feed = RealMarketFeed()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "chart": {
            "result": [
                {
                    "meta": {
                        "regularMarketPrice": 5025.5,
                        "chartPreviousClose": 5000.0,
                        "currency": "USD",
                        "exchangeTimezoneName": "America/New_York",
                    }
                }
            ]
        }
    }
    client_mock = AsyncMock()
    client_mock.get.return_value = mock_resp
    client_mock.__aenter__.return_value = client_mock
    client_mock.__aexit__.return_value = None

    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: client_mock)

    quote = await feed.fetch_real_quote("OTC_SPC")
    assert quote is not None
    assert quote["price"] == 5025.5
    assert quote["currency"] == "USD"


@pytest.mark.asyncio
async def test_fetch_real_candles_empty_result(monkeypatch):
    feed = RealMarketFeed()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"chart": {"result": []}}
    client_mock = AsyncMock()
    client_mock.get.return_value = mock_resp
    client_mock.__aenter__.return_value = client_mock
    client_mock.__aexit__.return_value = None

    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: client_mock)
    candles = await feed.fetch_real_candles("OTC_SPC")
    assert candles == []


@pytest.mark.asyncio
async def test_fetch_real_candles_break_and_none(monkeypatch):
    feed = RealMarketFeed()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "chart": {
            "result": [
                {
                    "timestamp": [1700000000, 1700000900, 1700001800],
                    "indicators": {
                        "quote": [
                            {
                                "open": [5000.0, None],
                                "high": [5015.0, 5020.0],
                                "low": [4995.0, 5005.0],
                                "close": [5010.0, 5018.0],
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
    candles = await feed.fetch_real_candles("OTC_SPC")
    assert len(candles) == 1
    assert candles[0]["open"] == 5000.0


@pytest.mark.asyncio
async def test_fetch_real_candles_exception(monkeypatch):
    feed = RealMarketFeed()
    client_mock = AsyncMock()
    client_mock.get.side_effect = RuntimeError("network down")
    client_mock.__aenter__.return_value = client_mock
    client_mock.__aexit__.return_value = None

    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: client_mock)
    candles = await feed.fetch_real_candles("OTC_SPC")
    assert candles == []


@pytest.mark.asyncio
async def test_fetch_real_quote_unknown_symbol():
    feed = RealMarketFeed()
    quote = await feed.fetch_real_quote("UNKNOWN_ASSET")
    assert quote is None


@pytest.mark.asyncio
async def test_fetch_real_quote_http_error(monkeypatch):
    feed = RealMarketFeed()
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    client_mock = AsyncMock()
    client_mock.get.return_value = mock_resp
    client_mock.__aenter__.return_value = client_mock
    client_mock.__aexit__.return_value = None

    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: client_mock)
    quote = await feed.fetch_real_quote("OTC_SPC")
    assert quote is None


@pytest.mark.asyncio
async def test_fetch_real_quote_empty_result(monkeypatch):
    feed = RealMarketFeed()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"chart": {"result": []}}
    client_mock = AsyncMock()
    client_mock.get.return_value = mock_resp
    client_mock.__aenter__.return_value = client_mock
    client_mock.__aexit__.return_value = None

    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: client_mock)
    quote = await feed.fetch_real_quote("OTC_SPC")
    assert quote is None


@pytest.mark.asyncio
async def test_fetch_real_quote_exception(monkeypatch):
    feed = RealMarketFeed()
    client_mock = AsyncMock()
    client_mock.get.side_effect = RuntimeError("timeout")
    client_mock.__aenter__.return_value = client_mock
    client_mock.__aexit__.return_value = None

    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: client_mock)
    quote = await feed.fetch_real_quote("OTC_SPC")
    assert quote is None
