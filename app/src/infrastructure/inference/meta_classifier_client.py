"""Cliente HTTP assincrono para o meta-classificador tabular de stacking."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
from typing import Any

import httpx

from src.application.services.log_dedupe import clear_log_channel, log_warning_if_changed
from src.application.services.meta_classifier_cross_symbol import META_FEATURE_DIM
from src.application.services.meta_classifier_features import (
    extract_meta_feature_vector,
    side_payoff_from_probability,
)
from src.infrastructure.inference.meta_classifier_types import (
    MetaPredictRequest,
    MetaPredictResponse,
    parse_meta_predict_response,
)


logger = logging.getLogger("AETH")
META_TIMEOUT_SEC = 1.0


def assert_meta_feature_vector_dim(feature_vector: list[float]) -> None:
    """Pre-flight: bloqueia envio HTTP se o vetor tabular nao tiver META_FEATURE_DIM=43."""
    length = len(feature_vector)
    if length != META_FEATURE_DIM:
        raise ValueError(f"Vetor tabular corrompido: local esperado 43, gerado {length}")


class _MetaFallbackLogState:
    """Estado de deduplicacao para logs de fallback do meta-regressor."""


_META_FALLBACK_LOG = _MetaFallbackLogState()


def meta_classifier_enabled(config: dict[str, Any]) -> bool:
    """Indica se o meta-regressor tabular esta habilitado."""
    infra = config.get("infra") if isinstance(config, dict) else {}
    chunk = infra.get("meta_classifier") if isinstance(infra, dict) else {}
    if not isinstance(chunk, dict):
        return False
    return bool(chunk.get("enabled", False))


def meta_classifier_http_url(config: dict[str, Any]) -> str:
    """Resolve URL HTTP do meta-regressor."""
    infra = config.get("infra") if isinstance(config, dict) else {}
    chunk = infra.get("meta_classifier") if isinstance(infra, dict) else {}
    if isinstance(chunk, dict) and chunk.get("http_url"):
        return str(chunk["http_url"]).rstrip("/")
    return os.getenv("AETHER_META_CLASSIFIER_HTTP", "http://localhost:8005").rstrip("/")


def meta_classifier_timeout(config: dict[str, Any]) -> float:
    """Resolve timeout em segundos para chamadas ao meta-regressor."""
    infra = config.get("infra") if isinstance(config, dict) else {}
    chunk = infra.get("meta_classifier") if isinstance(infra, dict) else {}
    if isinstance(chunk, dict) and chunk.get("timeout_seconds") is not None:
        return float(chunk["timeout_seconds"])
    return META_TIMEOUT_SEC


def build_persistent_http_client(base_url: str, timeout: float) -> httpx.AsyncClient:
    """Cria httpx.AsyncClient persistente com keep-alive e pool de conexoes."""
    return httpx.AsyncClient(
        base_url=base_url.rstrip("/"),
        timeout=httpx.Timeout(float(timeout)),
        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
    )


def reset_meta_classifier_fallback_dedupe() -> None:
    """Limpa deduplicacao de fallback para permitir um log por ciclo M5."""
    clear_log_channel(_META_FALLBACK_LOG, "meta_classifier_fallback")


def _emit_meta_classifier_fallback(exc: str) -> None:
    """Emite log estruturado de fallback quando o meta-regressor falha."""
    log_warning_if_changed(
        _META_FALLBACK_LOG,
        logger,
        "meta_classifier_fallback",
        exc,
        "META_CLASSIFIER_FALLBACK | %s | usando score TCN organico",
        exc,
    )


class MetaClassifierClient:
    """Cliente assincrono httpx para o servico meta-regressor tabular."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout: float,
        enabled: bool,
        http_client: httpx.AsyncClient | None = None,
        owns_http_client: bool = True,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = float(timeout)
        self._enabled = bool(enabled)
        self._owns_http_client = bool(owns_http_client)
        self._batch_fallback_exc: str | None = None
        if http_client is not None:
            self._client = http_client
        else:
            self._client = build_persistent_http_client(self._base_url, self._timeout)
            self._owns_http_client = True

    @property
    def enabled(self) -> bool:
        """Indica se chamadas remotas ao meta-regressor estao ativas."""
        return self._enabled

    def reset_batch_fallback_state(self) -> None:
        """Reinicia acumulador de falhas para emissao unica de log no batch."""
        self._batch_fallback_exc = None

    async def aclose(self) -> None:
        """Fecha o httpx.AsyncClient interno quando o pool e proprietario."""
        if self._owns_http_client:
            await self._client.aclose()

    async def predict_meta(
        self,
        request: MetaPredictRequest,
        *,
        fallback_score: float,
        defer_fallback_log: bool = False,
    ) -> MetaPredictResponse:
        """Consulta /v2/predict_meta retornando edge continuo de payoff."""
        if not self._enabled:
            return {"predicted_payoff_edge": 0.0, "meta_applied": False, "edge_expectancy": "LOSS_EXPECTED"}
        feature_vector = [float(v) for v in request["feature_vector"]]
        assert_meta_feature_vector_dim(feature_vector)
        payload = {
            "symbol": str(request["symbol"]),
            "tcn_probability": float(request["tcn_probability"]),
            "direction": str(request["direction"]),
            "feature_vector": feature_vector,
        }
        try:
            response = await self._client.post("/v2/predict_meta", json=payload)
            response.raise_for_status()
            parsed = parse_meta_predict_response(response.json())
            if parsed["meta_applied"]:
                reset_meta_classifier_fallback_dedupe()
            bb_width_z = float(feature_vector[8]) if len(feature_vector) > 8 else 0.0
            implied_vol = float(feature_vector[30]) if len(feature_vector) > 30 else 1.0
            if bb_width_z > 2.5 or implied_vol > 2.5:
                edge = parsed["predicted_payoff_edge"]
                if edge > 0.0:
                    parsed["predicted_payoff_edge"] = edge * 0.5
                else:
                    parsed["predicted_payoff_edge"] = edge * 1.5  # pragma: no cover
            return parsed
        except (httpx.TimeoutException, httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            _ = fallback_score
            message = str(exc)
            if defer_fallback_log:
                if self._batch_fallback_exc is None:
                    self._batch_fallback_exc = message
            else:
                _emit_meta_classifier_fallback(message)
            return {
                "predicted_payoff_edge": 0.0,
                "meta_applied": False,
                "edge_expectancy": "LOSS_EXPECTED",
            }  # pragma: no cover

    async def predict_meta_batch(
        self,
        requests: list[tuple[MetaPredictRequest, float]],
    ) -> list[MetaPredictResponse]:
        """Executa predicoes meta em paralelo com um unico log de fallback por ciclo."""
        self.reset_batch_fallback_state()
        reset_meta_classifier_fallback_dedupe()
        tasks = [self.predict_meta(req, fallback_score=fallback, defer_fallback_log=True) for req, fallback in requests]
        results = await asyncio.gather(*tasks)
        if self._batch_fallback_exc is not None:
            _emit_meta_classifier_fallback(self._batch_fallback_exc)
        return list(results)

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


def build_meta_predict_request(
    *,
    symbol: str,
    metrics: dict[str, Any],
    tcn_probability: float,
    direction: str,
) -> MetaPredictRequest:
    """Monta payload tipado para o endpoint meta-regressor."""
    return {
        "symbol": str(symbol),
        "tcn_probability": float(tcn_probability),
        "direction": str(direction),
        "feature_vector": extract_meta_feature_vector(metrics),
    }


def fallback_payoff_score(metrics: dict[str, Any], direction: str, tcn_probability: float) -> float:
    """Score bruto TCN usado como fallback quando o meta-regressor falha."""
    trade_score = metrics.get("trade_score")
    if trade_score is not None:
        return max(0.0, min(1.0, float(trade_score)))
    return side_payoff_from_probability(tcn_probability, direction)
