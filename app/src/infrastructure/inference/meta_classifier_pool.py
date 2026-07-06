"""Pool singleton e ciclo de vida do cliente HTTP meta-regressor."""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any

from src.infrastructure.inference.meta_classifier_client import (
    MetaClassifierClient,
    meta_classifier_enabled,
    meta_classifier_http_url,
    meta_classifier_timeout,
)
from src.infrastructure.inference.meta_classifier_types import MetaPredictRequest, MetaPredictResponse


_CLIENT_LOCK = asyncio.Lock()


class _MetaClientPool:
    """Pool singleton do cliente meta-classificador."""

    client: MetaClassifierClient | None = None


async def get_meta_classifier_client(config: dict[str, Any]) -> MetaClassifierClient:
    """Retorna singleton do cliente meta-regressor."""
    async with _CLIENT_LOCK:
        if _MetaClientPool.client is None:
            _MetaClientPool.client = MetaClassifierClient(
                base_url=meta_classifier_http_url(config),
                timeout=meta_classifier_timeout(config),
                enabled=meta_classifier_enabled(config),
            )
        return _MetaClientPool.client


async def bootstrap_meta_classifier_client(config: dict[str, Any]) -> MetaClassifierClient | None:
    """Inicializa cliente persistente no bootstrap da sessao de trading."""
    if not meta_classifier_enabled(config):
        return None
    return await get_meta_classifier_client(config)


async def close_meta_classifier_client() -> None:
    """Fecha cliente singleton do meta-regressor."""
    async with _CLIENT_LOCK:
        if _MetaClientPool.client is not None:
            await _MetaClientPool.client.aclose()
            _MetaClientPool.client = None


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
