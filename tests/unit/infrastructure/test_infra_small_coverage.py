"""Testes complementares para 100% de cobertura em state/storage/market."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.infrastructure.market.null_market_writer import NullMarketWriter
from src.infrastructure.state.json_state_store import JsonStateStore
from src.infrastructure.state.redis_state_pipeline import write_state_bundle
from src.infrastructure.storage.local_model_store import LocalModelStore
from src.infrastructure.storage.minio_model_store import MinioModelStore


@pytest.mark.asyncio
async def test_null_market_writer_flush_and_close():
    writer = NullMarketWriter()
    await writer.enqueue_tick(symbol="R_10", epoch_ms=1000, price=1.0)
    await writer.enqueue_bar(symbol="R_10", bar={"close": 1.0})
    await writer.enqueue_contract_audit({"contract_id": 1})
    await writer.record_quote_observation({"proposal_id": "p1"})
    await writer.flush()
    assert await writer.ping() is True
    await writer.close()


@pytest.mark.asyncio
async def test_json_state_store_all_branches(tmp_path):
    path = tmp_path / "state.json"
    store = JsonStateStore(path)
    assert await store.load_snapshot() is None
    assert store.load() is None

    await store.save_state_bundle(
        snapshot={"risk": {"initial_bankroll": 100.0, "pending_loss": {"R_10": 2.0}}},
        session={"k": "v"},
        market_sig="sig1",
        recovery_skip_counter=2,
        session_start_balance=100.0,
        session_target_win=10.0,
        dlambert_unit=1.0,
        consecutive_losses_linear=1,
    )
    assert await store.get_hash("state:pending_loss") == {"R_10": "2.0"}
    await store.delete_string("market_sig")
    assert await store.get_string("market_sig") is None
    await store.close()

    path.write_text("{corrupted", encoding="utf-8")
    assert await store.load_snapshot() is None
    assert store.load() is None


@pytest.mark.asyncio
async def test_redis_state_pipeline_all_optional_keys():
    client = MagicMock()
    pipe = MagicMock()
    pipe.__aenter__ = AsyncMock(return_value=pipe)
    pipe.__aexit__ = AsyncMock(return_value=None)
    pipe.execute = AsyncMock()
    client.pipeline.return_value = pipe
    await write_state_bundle(
        client,
        prefix="aether",
        snapshot={"risk": {}},
        recovery_skip_counter=1,
        session_start_balance=100.0,
        session_target_win=5.0,
        dlambert_unit=2.0,
        consecutive_losses_linear=1,
    )
    assert pipe.set.call_count >= 6


@pytest.mark.asyncio
async def test_local_and_minio_model_store_branches(tmp_path):
    local = LocalModelStore(tmp_path)
    ts_src = tmp_path / "R_10" / "tcn" / "latest_ts.pt"
    ts_src.parent.mkdir(parents=True, exist_ok=True)
    ts_src.write_bytes(b"ts")
    dest = tmp_path / "out" / "ts.pt"
    assert await local.download_torchscript("R_10", arch="tcn", dest=dest) is True
    assert await local.load_manifest("R_10", arch="tcn") == {}
    (tmp_path / "R_10" / "tcn" / "manifest.json").write_text("{bad", encoding="utf-8")
    assert await local.load_manifest("R_10", arch="tcn") == {}
    (tmp_path / "R_10" / "tcn" / "manifest.json").write_text('{"ok": true}', encoding="utf-8")
    assert await local.load_manifest("R_10", arch="tcn") == {"ok": True}
    await local.close()

    fake_client = MagicMock()
    fake_client.bucket_exists.return_value = True
    resp = MagicMock()
    resp.read.return_value = json.dumps({"v": 1}).encode("utf-8")
    fake_client.get_object.return_value = resp
    minio = MinioModelStore.__new__(MinioModelStore)
    minio._client = fake_client
    minio._bucket = "models"
    pth = tmp_path / "latest.pth"
    pth.write_bytes(b"pth")
    ts = tmp_path / "latest_ts.pt"
    ts.write_bytes(b"ts")
    await minio.upload("R_10", pth, arch="tcn", metadata={"v": 1})
    assert fake_client.fput_object.call_count == 2
    assert await minio.download_torchscript("R_10", arch="tcn", dest=tmp_path / "dl" / "ts.pt") is True
    fake_client.fget_object.side_effect = RuntimeError("fail")
    assert await minio.download_torchscript("R_10", arch="tcn", dest=tmp_path / "dl" / "ts2.pt") is False
    assert await minio.load_manifest("R_10", arch="tcn") == {"v": 1}
    fake_client.get_object.side_effect = RuntimeError("missing")
    assert await minio.load_manifest("R_10", arch="tcn") == {}


@pytest.mark.asyncio
async def test_infra_factory_history_stream_tick_and_redis_edges(monkeypatch):
    from datetime import UTC, datetime

    from src.domain.models.market_data import Candle
    from src.infrastructure.api.deriv_credentials import resolve_deriv_app_id
    from src.infrastructure.factories.infra_factory import InfraServices, _resolve_redis_url, validate_infra_services
    from src.infrastructure.handlers.history_fetch import is_rate_limit_error
    from src.infrastructure.handlers.stream_handler import StreamHandler
    from src.infrastructure.handlers.stream_timeframe import granularity_label, resolve_mini_fetch_count
    from src.infrastructure.handlers.tick_buffer import TickBuffer
    from src.infrastructure.state.redis_state_store import RedisStateStore

    monkeypatch.delenv("AETHER_REDIS_URL", raising=False)
    monkeypatch.delenv("REDIS_PORT", raising=False)
    assert _resolve_redis_url({}) == "redis://127.0.0.1:6379/0"
    monkeypatch.setenv("REDIS_PORT", "6380")
    assert _resolve_redis_url({"url": "redis://127.0.0.1:6379/0"}) == "redis://127.0.0.1:6380/0"
    monkeypatch.setenv("AETHER_REDIS_URL", "redis://127.0.0.1:6399/1")
    assert _resolve_redis_url({}) == "redis://127.0.0.1:6399/1"

    services_ok = InfraServices(
        enabled=True,
        fail_fast=False,
        state_store=SimpleNamespace(ping=AsyncMock(side_effect=TimeoutError("redis timeout")), close=AsyncMock()),
        market_writer=SimpleNamespace(ping=AsyncMock(return_value=False), flush=AsyncMock()),
        model_store=SimpleNamespace(head=AsyncMock(return_value=True)),
    )
    await validate_infra_services(services_ok, {"infra": {"fail_fast": False}})

    assert is_rate_limit_error({"error": "rate limit exceeded"}) is True
    monkeypatch.delenv("DERIV_APP_ID", raising=False)
    monkeypatch.delenv("AETHER_DERIV_APP_ID", raising=False)
    assert resolve_deriv_app_id(None, repo_root=Path(".nonexistent_repo_dir")) == ""
    assert resolve_mini_fetch_count({"mini_history_bars": 42}) == 42
    assert granularity_label(3600) == "H1"
    assert granularity_label(30) == "S30"

    tb = TickBuffer(["R_10"])
    assert tb.live_tick_count("UNKNOWN") == 0
    assert tb.forming_bar_micro_stats("R_10").tick_count == 0
    assert tb.microstructure_series("R_10", 0) == []

    sh = StreamHandler(MagicMock(), ["R_10"], {"micro_granularity": 300})
    sh.ws.send = AsyncMock(return_value={"candles": []})
    await sh.ensure_cluster_history(1, timeframe="mini")
    await sh._apply_mini_candle("UNKNOWN", Candle("UNKNOWN", 1.0, 1.0, 1.0, 1.0, datetime.now(UTC), 300))

    redis_store = RedisStateStore(url="redis://localhost:6379/0")
    redis_client = MagicMock(get=AsyncMock(return_value=b"{corrupted"))
    redis_store._redis = AsyncMock(return_value=redis_client)
    assert await redis_store.load_snapshot() is None
