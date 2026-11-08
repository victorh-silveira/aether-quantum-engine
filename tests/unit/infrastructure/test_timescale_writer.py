"""Testes do TimescaleMarketWriter com pool mockado."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.infrastructure.market.null_market_writer import NullMarketWriter
from src.infrastructure.market.timescale_writer import _CONTRACT_FIELDS, TimescaleMarketWriter


@pytest.mark.asyncio
async def test_contract_audit_defaults_missing_source_to_pending():
    writer = TimescaleMarketWriter(dsn="postgresql://u:p@localhost/db")
    with patch.object(writer, "_ensure_worker"):
        await writer.enqueue_contract_audit(
            {"contract_id": 42, "quote_price": 10.0, "quote_payout": 17.8, "quote_ev": 0.02}
        )
        _, values = writer._queue.get_nowait()
    assert values[_CONTRACT_FIELDS.index("settlement_source")] == "pending"
    assert values[_CONTRACT_FIELDS.index("quote_price")] == 10.0
    assert values[_CONTRACT_FIELDS.index("quote_payout")] == 17.8
    assert values[_CONTRACT_FIELDS.index("quote_ev")] == 0.02


@pytest.mark.asyncio
async def test_null_market_writer_noop():
    writer = NullMarketWriter()
    await writer.enqueue_tick(symbol="R_10", epoch_ms=1, price=1.0)
    await writer.enqueue_bar(symbol="R_10", bar={"epoch": 1})
    await writer.enqueue_contract_audit({"contract_id": 1})
    await writer.record_quote_observation({"symbol": "R_25"})
    assert await writer.ping() is True


@pytest.mark.asyncio
async def test_timescale_writer_enqueue_and_flush():
    writer = TimescaleMarketWriter(dsn="postgresql://u:p@localhost/db", flush_interval_ms=1.0, batch_limit=2)
    conn = AsyncMock()
    conn.executemany = AsyncMock()
    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    with patch.object(writer, "_ensure_pool", AsyncMock(return_value=pool)):
        await writer.enqueue_tick(symbol="R_10", epoch_ms=1000, price=1.23)
        await writer.enqueue_bar(
            symbol="R_10",
            bar={
                "epoch": 1,
                "granularity": 900,
                "open": 1.0,
                "high": 2.0,
                "low": 0.5,
                "close": 1.5,
                "tick_count": 3,
                "mean_inter_tick_ms": 10.0,
                "price_velocity": 0.1,
            },
        )
        await writer.flush()
    assert conn.executemany.await_count >= 1


@pytest.mark.asyncio
async def test_timescale_writer_contract_batch_and_closed():
    writer = TimescaleMarketWriter(dsn="postgresql://u:p@localhost/db", flush_interval_ms=1.0)
    conn = AsyncMock()
    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    with patch.object(writer, "_ensure_pool", AsyncMock(return_value=pool)):
        await writer.enqueue_contract_audit({"contract_id": 7, "symbol": "1HZ75V"})
        await writer.flush()
    assert any("contract_executions" in call.args[0] for call in conn.executemany.await_args_list)
    await writer.close()
    await writer.enqueue_contract_audit({"contract_id": 8})
    assert writer._queue.empty()


@pytest.mark.asyncio
async def test_timescale_writer_exposes_failed_batch_to_pilot():
    writer = TimescaleMarketWriter(dsn="postgresql://u:p@localhost/db", flush_interval_ms=1.0)
    with patch.object(writer, "_flush_batches", AsyncMock(side_effect=RuntimeError("schema ausente"))):
        await writer.enqueue_contract_audit({"contract_id": 7, "symbol": "R_25"})
        await writer._worker
    assert isinstance(writer._last_error, RuntimeError)


@pytest.mark.asyncio
async def test_timescale_writer_manual_flush_contract_when_worker_not_started():
    writer = TimescaleMarketWriter(dsn="postgresql://u:p@localhost/db")
    conn = AsyncMock()
    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    with (
        patch.object(writer, "_ensure_worker"),
        patch.object(writer, "_ensure_pool", AsyncMock(return_value=pool)),
    ):
        await writer.enqueue_contract_audit({"contract_id": 12, "symbol": "1HZ75V"})
        await writer.flush()
    assert any("contract_executions" in call.args[0] for call in conn.executemany.await_args_list)


@pytest.mark.asyncio
async def test_quote_observation_is_written_before_buy_and_rejects_closed_writer():
    writer = TimescaleMarketWriter(dsn="postgresql://u:p@localhost/db")
    conn = AsyncMock()
    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    row = dict.fromkeys(
        (
            "account_mode",
            "symbol",
            "period_seconds",
            "candle_epoch",
            "model_version",
            "side",
            "proposal_id",
            "side_probability",
            "quote_price",
            "quote_payout",
            "quote_ev",
            "cohort",
        ),
        1,
    )
    with patch.object(writer, "_ensure_pool", AsyncMock(return_value=pool)):
        await writer.record_quote_observation(row)
    assert "indicator_quote_observations" in conn.execute.await_args.args[0]
    writer._closed = True
    with pytest.raises(RuntimeError, match="encerrada"):
        await writer.record_quote_observation(row)


@pytest.mark.asyncio
async def test_timescale_ping():
    writer = TimescaleMarketWriter(dsn="postgresql://u:p@localhost/db")
    conn = AsyncMock()
    conn.fetchval = AsyncMock(return_value=1)
    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    with patch.object(writer, "_ensure_pool", AsyncMock(return_value=pool)):
        assert await writer.ping() is True
