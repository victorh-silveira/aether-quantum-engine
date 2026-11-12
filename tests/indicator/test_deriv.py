"""Fonte publica deve impedir operacoes de conta e contratos."""

import asyncio
import json

import pytest

from indicator.deriv import PublicDeriv


@pytest.mark.asyncio
async def test_public_client_only_market_data():
    payloads = []

    async def request(payload):
        payloads.append(payload)
        return {"active_symbols": [{"market": "synthetic_index"}], "candles": [{"epoch": 300}]}

    client = PublicDeriv(request=request)
    assert await client.symbols() == [{"market": "synthetic_index"}]
    assert await client.candles("R_75") == [{"epoch": 300}]
    assert await client.candles("R_75", period=900) == [{"epoch": 300}]
    assert payloads[-1]["granularity"] == 900
    with pytest.raises(ValueError, match="periodo nao suportado"):
        await client.candles("R_75", period=60)
    assert all("buy" not in payload and "authorize" not in payload for payload in payloads)
    with pytest.raises(ValueError):
        await client.request({"buy": "123"})
    await client.close()


@pytest.mark.asyncio
async def test_socket_reconnect_and_error(monkeypatch):
    class Socket:
        def __init__(self):
            self.sent = []
            self.closed = False

        async def send(self, data):
            self.sent.append(json.loads(data))

        async def recv(self):
            return json.dumps({"req_id": self.sent[-1]["req_id"], "active_symbols": []})

        async def close(self):
            self.closed = True

    socket = Socket()

    async def connect(*args, **kwargs):
        return socket

    monkeypatch.setattr("indicator.deriv.websockets.connect", connect)
    client = PublicDeriv()
    assert await client.symbols() == []
    await client.close()
    assert socket.closed

    async def timeout(*args, **kwargs):
        raise TimeoutError

    monkeypatch.setattr("indicator.deriv.websockets.connect", timeout)
    with pytest.raises(asyncio.TimeoutError):
        await client.symbols()


@pytest.mark.asyncio
async def test_socket_response_validation(monkeypatch):
    class Socket:
        async def send(self, data):
            pass

        async def recv(self):
            return json.dumps(self.response)

        async def close(self):
            pass

    socket = Socket()

    async def connect(*args, **kwargs):
        return socket

    monkeypatch.setattr("indicator.deriv.websockets.connect", connect)
    client = PublicDeriv()
    socket.response = {"error": {"message": "rejected"}}
    with pytest.raises(RuntimeError, match="rejected"):
        await client.symbols()
    socket.response = {"req_id": 999}
    with pytest.raises(RuntimeError, match="fora de ordem"):
        await client.symbols()


@pytest.mark.asyncio
async def test_public_client_rtt_callback_success_and_failure():
    events: list[tuple[str, float, bool]] = []

    async def ok_req(payload):
        return {"active_symbols": []}

    client_ok = PublicDeriv(request=ok_req, on_rtt=lambda m, ms, *, ok=True: events.append((m, ms, ok)))
    await client_ok.symbols()
    assert len(events) == 1
    assert events[0][0] == "active_symbols"
    assert events[0][2] is True

    async def fail_req(payload):
        raise RuntimeError("boom")

    client_err = PublicDeriv(request=fail_req, on_rtt=lambda m, ms, *, ok=True: events.append((m, ms, ok)))
    with pytest.raises(RuntimeError, match="boom"):
        await client_err.candles("OTC_SSMI", period=900)
    assert events[-1][0] == "ticks_history"
    assert events[-1][2] is False
