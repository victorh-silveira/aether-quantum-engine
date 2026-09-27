"""Entradas operacionais Touch: sem sanitize destrutivo nem ordens para coletar dados."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.operations import (
    collect_touch_dataset as collect,
    train_touch_classifier as train,
)


@pytest.mark.asyncio
async def test_train_dispatch_and_data_errors(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train_touch_classifier.py", "--if-enabled"])
    with patch.object(train, "load_settings_json", return_value={}):
        assert await train.main() == 3
    settings = {"touch": {"enabled": True}, "infra": {"timescale": {"dsn": "dsn"}}}
    with (
        patch.object(train, "load_settings_json", return_value=settings),
        patch.object(train, "read_touch_quotes", return_value=[]),
    ):
        assert await train.main() == 2
    quote = {"group_ms": 10000, "decision_ms": 10000}
    with (
        patch.object(train, "load_settings_json", return_value=settings),
        patch.object(train, "read_touch_quotes", return_value=[quote]),
        patch.object(train, "read_touch_ticks", AsyncMock(return_value=[])),
        patch.object(train, "build_touch_samples", return_value=([], 2)),
        patch.object(train, "train_touch_model", return_value={"qualified": False, "oos": {}}),
        patch.object(train, "save_touch_model", return_value="candidate.json") as save,
    ):
        assert await train.main() == 0
        assert save.call_args.args[0]["qualified"] is False


@pytest.mark.asyncio
async def test_collector_only_subscribes_and_quotes():
    ws = MagicMock(is_running=True, subscriptions={})
    ws.connect = AsyncMock()
    ws.close = AsyncMock()

    async def send(_request):
        await ws.subscriptions["tick"]({"tick": {"symbol": "1HZ75V", "epoch": 1, "quote": 100}})
        await ws.subscriptions["tick"]({"msg_type": "ping"})
        return {}

    ws.send = AsyncMock(side_effect=send)
    writer = MagicMock(ping=AsyncMock(return_value=True), close=AsyncMock(), enqueue_tick=AsyncMock())
    with (
        patch.object(collect, "WebSocketManager", return_value=ws),
        patch.object(collect, "TimescaleMarketWriter", return_value=writer),
        patch.object(collect, "monotonic", side_effect=[0, 301, 301]),
        patch.object(collect, "quote_touch_candidates", AsyncMock(return_value=[{"proposal_id": "q"}])),
        patch.object(collect, "append_touch_quotes") as append,
        patch.object(collect.asyncio, "sleep", AsyncMock()),
    ):
        assert await collect.collect_dataset({"infra": {"timescale": {"dsn": "dsn"}}}, max_groups=1) == 1
    ws.send.assert_awaited_once_with({"ticks": "1HZ75V", "subscribe": 1})
    writer.enqueue_tick.assert_awaited_once()
    append.assert_called_once()
    writer.close.assert_awaited_once()
    ws.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_collector_closes_when_db_unavailable():
    ws = MagicMock(close=AsyncMock())
    writer = MagicMock(ping=AsyncMock(return_value=False), close=AsyncMock())
    with (
        patch.object(collect, "WebSocketManager", return_value=ws),
        patch.object(collect, "TimescaleMarketWriter", return_value=writer),
        pytest.raises(ConnectionError),
    ):
        await collect.collect_dataset({"infra": {"timescale": {"dsn": "dsn"}}}, max_groups=1)
    ws.close.assert_awaited_once()
    writer.close.assert_awaited_once()
