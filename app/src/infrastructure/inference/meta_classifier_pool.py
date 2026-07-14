"""Pool singleton e ciclo de vida do cliente HTTP meta-regressor."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import threading
from typing import Any

from src.infrastructure.inference.meta_classifier_client import (
    MetaClassifierClient,
    meta_classifier_enabled,
    meta_classifier_http_url,
    meta_classifier_timeout,
)
from src.infrastructure.inference.meta_classifier_types import MetaPredictRequest, MetaPredictResponse


_CLIENT_GUARD = threading.Lock()


class _MetaClientPool:
    """Pool singleton do cliente meta-classificador."""

    client: MetaClassifierClient | None = None
    loop: asyncio.AbstractEventLoop | None = None


async def get_meta_classifier_client(config: dict[str, Any]) -> MetaClassifierClient:
    """Retorna singleton do cliente meta-regressor no event loop corrente."""
    loop = asyncio.get_running_loop()
    stale: MetaClassifierClient | None = None
    with _CLIENT_GUARD:
        client = _MetaClientPool.client
        if client is not None and _MetaClientPool.loop is not loop:
            stale = client
            _MetaClientPool.client = None
            _MetaClientPool.loop = None
            client = None
        if client is None:
            client = MetaClassifierClient(
                base_url=meta_classifier_http_url(config),
                timeout=meta_classifier_timeout(config),
                enabled=meta_classifier_enabled(config),
            )
            _MetaClientPool.client = client
            _MetaClientPool.loop = loop
    if stale is not None:
        with contextlib.suppress(Exception):
            await stale.aclose()
    return client


async def bootstrap_meta_classifier_client(config: dict[str, Any]) -> MetaClassifierClient | None:
    """Inicializa cliente persistente no bootstrap da sessao de trading."""
    if not meta_classifier_enabled(config):
        return None
    return await get_meta_classifier_client(config)


async def close_meta_classifier_client() -> None:
    """Fecha cliente singleton do meta-regressor."""
    with _CLIENT_GUARD:
        client = _MetaClientPool.client
        _MetaClientPool.client = None
        _MetaClientPool.loop = None
    if client is not None:
        with contextlib.suppress(Exception):
            await client.aclose()


def predict_meta_via_config_sync(
    config: dict[str, Any],
    request: MetaPredictRequest,
    *,
    fallback_score: float,
) -> MetaPredictResponse:
    """Executa predicao meta reutilizando o singleton HTTP persistente."""

    async def _run() -> MetaPredictResponse:
        """Executa predicao com cliente singleton compartilhado."""
        client = await get_meta_classifier_client(config)
        return await client.predict_meta(request, fallback_score=fallback_score)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_run())
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, _run()).result(timeout=meta_classifier_timeout(config) + 0.5)
