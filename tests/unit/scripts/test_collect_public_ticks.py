"""Captura publica nao usa trading nem aceita outro simbolo."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from scripts.operations.collect_public_ticks import collect_ticks, parse_tick


def test_parse_tick_only_current_symbol_and_valid_quote():
    assert parse_tick({"tick": {"symbol": "1HZ75V", "epoch": 100, "quote": "10.5"}}) == (100000, 10.5)
    assert parse_tick({"tick": {"symbol": "R_10", "epoch": 100, "quote": 10}}) is None
    assert parse_tick({"tick": {"symbol": "1HZ75V", "epoch": 0, "quote": 10}}) is None
    assert parse_tick({"tick": {"symbol": "1HZ75V", "epoch": 1, "quote": "nan"}}) is None
    assert parse_tick({"tick": {"symbol": "1HZ75V", "epoch": "bad", "quote": 10}}) is None


@pytest.mark.asyncio
async def test_collect_ticks_subscribes_public_only_and_stops_at_limit():
    class Socket:
        def __init__(self):
            self.send = AsyncMock()
            self.close = AsyncMock()

        def __aiter__(self):
            return self._rows()

        async def _rows(self):
            for row in [
                {"msg_type": "subscription"},
                {"tick": {"symbol": "1HZ75V", "epoch": 100, "quote": 1.0}},
                {"tick": {"symbol": "1HZ75V", "epoch": 101, "quote": 2.0}},
            ]:
                yield json.dumps(row)

    socket = Socket()
    writer = AsyncMock()
    with patch("scripts.operations.collect_public_ticks.connect_wss_with_ip_failover", AsyncMock(return_value=socket)):
        n = await collect_ticks(writer, "wss://public", max_ticks=2)
    assert n == 2
    assert json.loads(socket.send.call_args.args[0]) == {"ticks": "1HZ75V", "subscribe": 1}
    assert writer.enqueue_tick.await_count == 2
    writer.flush.assert_awaited_once()
    socket.close.assert_awaited_once()
