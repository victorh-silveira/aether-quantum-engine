"""Fabrica de servicos de infraestrutura com validacao fail-fast."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from src.infrastructure.market.null_market_writer import NullMarketWriter
from src.infrastructure.market.timescale_writer import TimescaleMarketWriter
from src.infrastructure.state.json_state_store import JsonStateStore
from src.infrastructure.state.redis_state_store import RedisStateStore
from src.infrastructure.storage.local_model_store import LocalModelStore
from src.infrastructure.storage.minio_model_store import MinioModelStore


@dataclass
class InfraServices:
    """Bundle de portas de infraestrutura."""

    enabled: bool
    fail_fast: bool
    state_store: Any
    market_writer: Any
    model_store: Any


def _infra_cfg(config: dict[str, Any]) -> dict[str, Any]:
    """Extrai bloco infra da configuracao raiz."""
    chunk = config.get("infra") if isinstance(config, dict) else {}
    return chunk if isinstance(chunk, dict) else {}


def create_infra_services(config: dict[str, Any]) -> InfraServices:
    """Instancia stores conforme bloco infra da configuracao."""
    cfg = _infra_cfg(config)
    enabled = bool(cfg.get("enabled", False))
    fail_fast = bool(cfg.get("fail_fast", True))
    if not enabled:
        return InfraServices(
            enabled=False,
            fail_fast=False,
            state_store=JsonStateStore(),
            market_writer=NullMarketWriter(),
            model_store=LocalModelStore(),
        )
    redis_cfg = cfg.get("redis") if isinstance(cfg.get("redis"), dict) else {}
    ts_cfg = cfg.get("timescale") if isinstance(cfg.get("timescale"), dict) else {}
    minio_cfg = cfg.get("minio") if isinstance(cfg.get("minio"), dict) else {}
    state_store = RedisStateStore(
        url=str(redis_cfg.get("url", "redis://127.0.0.1:6379/0")),
        key_prefix=str(redis_cfg.get("key_prefix", "aether")),
        socket_connect_timeout=float(redis_cfg.get("socket_connect_timeout_seconds", 2.0)),
        socket_timeout=float(redis_cfg.get("socket_timeout_seconds", 15.0)),
    )
    market_writer = TimescaleMarketWriter(
        dsn=str(ts_cfg.get("dsn", "postgresql://aether:aether@localhost:5432/aether")),
    )
    model_store = MinioModelStore(
        endpoint=str(minio_cfg.get("endpoint", "localhost:9000")),
        bucket=str(minio_cfg.get("bucket", "dl-models")),
        access_key=os.getenv("AETHER_MINIO_ACCESS_KEY", "aether"),
        secret_key=os.getenv("AETHER_MINIO_SECRET_KEY", "aethersecret"),
        secure=bool(minio_cfg.get("secure", False)),
    )
    return InfraServices(
        enabled=True,
        fail_fast=fail_fast,
        state_store=state_store,
        market_writer=market_writer,
        model_store=model_store,
    )


async def validate_infra_services(services: InfraServices, config: dict[str, Any]) -> None:
    """Valida conectividade Redis, Timescale e MinIO antes do startup."""
    logger = logging.getLogger("AETH")
    if not services.enabled:
        return
    cfg = _infra_cfg(config)
    fail_fast = bool(cfg.get("fail_fast", True)) and services.fail_fast
    checks = [
        ("Redis", services.state_store.ping()),
        ("TimescaleDB", services.market_writer.ping()),
        ("MinIO", services.model_store.head()),
    ]
    for label, coro in checks:
        ok = await coro
        if ok:
            logger.debug("INFRA: %s ok", label)
            continue
        message = f"INFRA: {label} indisponivel em localhost; make docker-up-core|docker-up|docker-up-cpu"
        if fail_fast:
            raise ConnectionError(message)
        logger.warning(message)


async def close_infra_services(services: InfraServices) -> None:
    """Encerra pools e conexoes de infraestrutura."""
    await services.market_writer.flush()
    await services.state_store.close()
    await services.market_writer.close()
    await services.model_store.close()
