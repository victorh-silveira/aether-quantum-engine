"""Cliente HTTP assincrono para o meta-classificador tabular de stacking."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
from typing import Any

import httpx

from src.application.services.meta_classifier_features import (
    extract_meta_feature_vector,
    side_payoff_from_probability,
)
from src.infrastructure.inference.meta_classifier_types import MetaPredictRequest, MetaPredictResponse


logger = logging.getLogger("AETH")
META_TIMEOUT_SEC = 1.0
_CLIENT_LOCK = asyncio.Lock()


class _MetaClientPool:
    """Pool singleton do cliente meta-classificador."""

    client: MetaClassifierClient | None = None


def meta_classifier_enabled(config: dict[str, Any]) -> bool:
    """Indica se o meta-classificador tabular esta habilitado."""
    infra = config.get("infra") if isinstance(config, dict) else {}
    chunk = infra.get("meta_classifier") if isinstance(infra, dict) else {}
    if not isinstance(chunk, dict):
        return False
    return bool(chunk.get("enabled", False))


def meta_classifier_http_url(config: dict[str, Any]) -> str:
    """Resolve URL HTTP do meta-classificador."""
    infra = config.get("infra") if isinstance(config, dict) else {}
    chunk = infra.get("meta_classifier") if isinstance(infra, dict) else {}
    if isinstance(chunk, dict) and chunk.get("http_url"):
        return str(chunk["http_url"]).rstrip("/")
    return os.getenv("AETHER_META_CLASSIFIER_HTTP", "http://localhost:8005").rstrip("/")


def meta_classifier_timeout(config: dict[str, Any]) -> float:
    """Resolve timeout em segundos para chamadas ao meta-classificador."""
    infra = config.get("infra") if isinstance(config, dict) else {}
    chunk = infra.get("meta_classifier") if isinstance(infra, dict) else {}
    if isinstance(chunk, dict) and chunk.get("timeout_seconds") is not None:
        return float(chunk["timeout_seconds"])
    return META_TIMEOUT_SEC


class MetaClassifierClient:
    """Cliente assincrono httpx para o servico meta-classificador tabular."""

    def __init__(self, *, base_url: str, timeout: float, enabled: bool) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = float(timeout)
        self._enabled = bool(enabled)
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout)

    @property
    def enabled(self) -> bool:
        """Indica se chamadas remotas ao meta-classificador estao ativas."""
        return self._enabled

    async def aclose(self) -> None:
        """Fecha o httpx.AsyncClient interno."""
        await self._client.aclose()

    async def predict_meta(
        self,
        request: MetaPredictRequest,
        *,
        fallback_score: float,
    ) -> MetaPredictResponse:
        """Consulta /v2/predict_meta com fallback transparente ao score TCN bruto."""
        if not self._enabled:
            return {"calibrated_payoff_score": float(fallback_score), "meta_applied": False}
        payload = {
            "symbol": str(request["symbol"]),
            "tcn_probability": float(request["tcn_probability"]),
            "direction": str(request["direction"]),
            "feature_vector": [float(v) for v in request["feature_vector"]],
        }
        try:
            response = await self._client.post("/v2/predict_meta", json=payload)
            response.raise_for_status()
            data = response.json()
            score = float(data["calibrated_payoff_score"])
            applied = bool(data.get("meta_applied", True))
            return {
                "calibrated_payoff_score": max(0.0, min(1.0, score)),
                "meta_applied": applied,
            }
        except (httpx.TimeoutException, httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            logger.warning("META_CLASSIFIER_FALLBACK | %s | usando trade_score TCN", exc)
            return {"calibrated_payoff_score": float(fallback_score), "meta_applied": False}

    async def predict_meta_batch(
        self,
        requests: list[tuple[MetaPredictRequest, float]],
    ) -> list[MetaPredictResponse]:
        """Executa predicoes meta em paralelo para multiplos simbolos."""
        tasks = [self.predict_meta(req, fallback_score=fallback) for req, fallback in requests]
        return await asyncio.gather(*tasks)

    def predict_meta_sync(
        self,
        request: MetaPredictRequest,
        *,
        fallback_score: float,
    ) -> MetaPredictResponse:
        """Ponte sincrona para o resolver quando nao ha prefetch assincrono."""

        async def _run() -> MetaPredictResponse:
            """Executa predict_meta no loop efemero da ponte sync."""
            return await self.predict_meta(request, fallback_score=fallback_score)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_run())
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, _run())
            return future.result(timeout=self._timeout + 0.5)


async def get_meta_classifier_client(config: dict[str, Any]) -> MetaClassifierClient:
    """Retorna singleton do cliente meta-classificador."""
    async with _CLIENT_LOCK:
        if _MetaClientPool.client is None:
            _MetaClientPool.client = MetaClassifierClient(
                base_url=meta_classifier_http_url(config),
                timeout=meta_classifier_timeout(config),
                enabled=meta_classifier_enabled(config),
            )
        return _MetaClientPool.client


async def close_meta_classifier_client() -> None:
    """Fecha cliente singleton do meta-classificador."""
    async with _CLIENT_LOCK:
        if _MetaClientPool.client is not None:
            await _MetaClientPool.client.aclose()
            _MetaClientPool.client = None


def build_meta_predict_request(
    *,
    symbol: str,
    metrics: dict[str, Any],
    tcn_probability: float,
    direction: str,
) -> MetaPredictRequest:
    """Monta payload tipado para o endpoint meta-classificador."""
    return {
        "symbol": str(symbol),
        "tcn_probability": float(tcn_probability),
        "direction": str(direction),
        "feature_vector": extract_meta_feature_vector(metrics),
    }


def fallback_payoff_score(metrics: dict[str, Any], direction: str, tcn_probability: float) -> float:
    """Score bruto TCN usado como fallback quando o meta-classificador falha."""
    trade_score = metrics.get("trade_score")
    if trade_score is not None:
        return max(0.0, min(1.0, float(trade_score)))
    return side_payoff_from_probability(tcn_probability, direction)


def predict_meta_via_config_sync(
    config: dict[str, Any],
    request: MetaPredictRequest,
    *,
    fallback_score: float,
) -> MetaPredictResponse:
    """Executa predicao meta com cliente efemero fora do singleton asyncio."""

    async def _run() -> MetaPredictResponse:
        """Executa predicao com cliente efemero e encerra o httpx.AsyncClient."""
        client = MetaClassifierClient(
            base_url=meta_classifier_http_url(config),
            timeout=meta_classifier_timeout(config),
            enabled=meta_classifier_enabled(config),
        )
        try:
            return await client.predict_meta(request, fallback_score=fallback_score)
        finally:
            await client.aclose()

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_run())
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, _run()).result(timeout=meta_classifier_timeout(config) + 0.5)
