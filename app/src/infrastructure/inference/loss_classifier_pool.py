"""Pool singleton do cliente loss-classifier."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import threading
from typing import Any

from src.infrastructure.inference.loss_classifier_client import (
    LossClassifierClient,
    build_loss_classifier_client_from_config,
    loss_classifier_enabled,
)
from src.infrastructure.inference.loss_classifier_types import LossPredictRequest, LossPredictResponse


_CLIENT_GUARD = threading.Lock()


class _LossClientPool:
    """Estado do singleton HTTP loss-classifier."""

    client: LossClassifierClient | None = None
    loop: asyncio.AbstractEventLoop | None = None


async def get_loss_classifier_client(config: dict[str, Any]) -> LossClassifierClient:
    """Retorna singleton no event loop corrente."""
    loop = asyncio.get_running_loop()
    stale: LossClassifierClient | None = None
    with _CLIENT_GUARD:
        client = _LossClientPool.client
        if client is not None and _LossClientPool.loop is not loop:
            stale = client
            _LossClientPool.client = None
            _LossClientPool.loop = None
            client = None
        if client is None:
            client = build_loss_classifier_client_from_config(config)
            _LossClientPool.client = client
            _LossClientPool.loop = loop
    if stale is not None:
        with contextlib.suppress(Exception):
            await stale.aclose()
    return client


async def close_loss_classifier_client() -> None:
    """Fecha singleton."""
    with _CLIENT_GUARD:
        client = _LossClientPool.client
        _LossClientPool.client = None
        _LossClientPool.loop = None
    if client is not None:
        with contextlib.suppress(Exception):
            await client.aclose()


def predict_loss_via_config_sync(config: dict[str, Any], request: LossPredictRequest) -> LossPredictResponse:
    """Predicao sincrona reutilizando o pool (thread/loop auxiliar)."""

    async def _run() -> LossPredictResponse:
        """Prediz via cliente do pool."""
        client = await get_loss_classifier_client(config)
        return await client.predict_loss(request)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_run())
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(_run())).result(timeout=5.0)


def learn_loss_via_config_sync(
    config: dict[str, Any],
    *,
    feature_vector: list[float],
    label: str,
    contract_id: str = "",
    symbol: str = "",
) -> dict[str, Any]:
    """Learn sincrona fail-open."""

    async def _run() -> dict[str, Any]:
        """Aprende sample WIN/LOSS via cliente do pool."""
        if not loss_classifier_enabled(config):
            return {"ok": False, "skipped": True}
        client = await get_loss_classifier_client(config)
        return await client.learn(
            feature_vector=feature_vector,
            label=label,
            contract_id=contract_id,
            symbol=symbol,
        )

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_run())
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(_run())).result(timeout=5.0)
