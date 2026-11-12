"""Rastreamento distribuido OTel com W3C TraceContext e exportacao OTLP/HTTP JSON."""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any
from uuid import uuid4

import httpx


_logger = logging.getLogger("AETH")
_TRACE_ID: ContextVar[str | None] = ContextVar("aether_otel_trace_id", default=None)
_SPAN_ID: ContextVar[str | None] = ContextVar("aether_otel_span_id", default=None)
_SPAN_BUFFER: deque[dict[str, Any]] = deque(maxlen=256)


def current_trace_id() -> str | None:
    """Retorna o trace_id hex de 32 caracteres ativo no contexto."""
    return _TRACE_ID.get()


def current_span_id() -> str | None:
    """Retorna o span_id hex de 16 caracteres ativo no contexto."""
    return _SPAN_ID.get()


def current_traceparent() -> str | None:
    """Formata cabecalho W3C traceparent do span ativo, quando existente."""
    tid = _TRACE_ID.get()
    sid = _SPAN_ID.get()
    if not tid or not sid:
        return None
    return f"00-{tid}-{sid}-01"


def _to_otlp_attr(key: str, value: Any) -> dict[str, Any]:
    """Converte par chave-valor em atributo compativel com OTLP JSON."""
    if isinstance(value, bool):
        return {"key": str(key), "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": str(key), "value": {"intValue": str(value)}}
    if isinstance(value, float):
        return {"key": str(key), "value": {"doubleValue": value}}
    return {"key": str(key), "value": {"stringValue": str(value)}}


@contextmanager
def trace_span(
    name: str,
    *,
    attributes: dict[str, Any] | None = None,
    service_name: str = "aether-engine",
) -> Iterator[str]:
    """Cria span hierarquico no ContextVar atual e registra payload OTLP em memoria."""
    parent_trace = _TRACE_ID.get()
    parent_span = _SPAN_ID.get()
    trace_id = parent_trace or uuid4().hex
    span_id = uuid4().hex[:16]
    tok_t = _TRACE_ID.set(trace_id)
    tok_s = _SPAN_ID.set(span_id)
    start_ns = time.time_ns()
    status = {"code": 1}
    attrs = dict(attributes or {})
    try:
        yield trace_id
    except Exception as exc:
        status = {"code": 2, "message": f"{type(exc).__name__}: {exc}"}
        raise
    finally:
        end_ns = max(start_ns + 1, time.time_ns())
        span_row: dict[str, Any] = {
            "traceId": trace_id,
            "spanId": span_id,
            "name": str(name),
            "kind": 1,
            "startTimeUnixNano": str(start_ns),
            "endTimeUnixNano": str(end_ns),
            "attributes": [_to_otlp_attr(k, v) for k, v in attrs.items() if v is not None],
            "status": status,
            "service_name": service_name,
        }
        if parent_span:
            span_row["parentSpanId"] = parent_span
        _SPAN_BUFFER.append(span_row)
        _TRACE_ID.reset(tok_t)
        _SPAN_ID.reset(tok_s)


def build_otlp_payload(spans: list[dict[str, Any]]) -> dict[str, Any]:
    """Agrupa spans por service_name no formato OTLP ExportTraceServiceRequest JSON."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in spans:
        svc = str(row.get("service_name") or "aether-engine")
        clean = {k: v for k, v in row.items() if k != "service_name"}
        grouped.setdefault(svc, []).append(clean)
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": [_to_otlp_attr("service.name", svc)]},
                "scopeSpans": [{"scope": {"name": "aether.otel", "version": "1.0.0"}, "spans": items}],
            }
            for svc, items in grouped.items()
        ]
    }


def _resolve_otlp_traces_url(endpoint: str | None) -> str | None:
    """Resolve URL final /v1/traces a partir de argumento ou variaveis de ambiente."""
    raw = (
        endpoint
        or os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    )
    if not raw or not str(raw).strip():
        return None
    url = str(raw).strip().rstrip("/")
    return url if url.endswith("/v1/traces") else f"{url}/v1/traces"


def drain_buffered_spans() -> list[dict[str, Any]]:
    """Drena e retorna todos os spans atualmente em buffer."""
    items = list(_SPAN_BUFFER)
    _SPAN_BUFFER.clear()
    return items


def flush_otlp_spans(endpoint: str | None = None, *, timeout: float = 1.5) -> int:
    """Envia spans pendentes ao coletor OTLP/HTTP JSON sem propagar falhas de rede."""
    spans = drain_buffered_spans()
    if not spans:
        return 0
    url = _resolve_otlp_traces_url(endpoint)
    if not url:
        return 0
    payload = build_otlp_payload(spans)
    try:
        resp = httpx.post(url, json=payload, timeout=timeout)
        if resp.status_code >= 400:
            _logger.debug("OTEL: exportacao HTTP %d descartada em %s", resp.status_code, url)
            return 0
        return len(spans)
    except Exception as exc:
        _logger.debug("OTEL: falha best-effort ao exportar %d spans: %s", len(spans), exc)
        return 0


async def flush_otlp_spans_async(endpoint: str | None = None, *, timeout: float = 1.5) -> int:
    """Versao assincrona nao bloqueante de exportacao OTLP/HTTP JSON."""
    url = _resolve_otlp_traces_url(endpoint)
    if not url:
        drain_buffered_spans()
        return 0
    spans = drain_buffered_spans()
    if not spans:
        return 0
    payload = build_otlp_payload(spans)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload)
        if resp.status_code >= 400:
            _logger.debug("OTEL: exportacao async HTTP %d descartada em %s", resp.status_code, url)
            return 0
        return len(spans)
    except Exception as exc:
        _logger.debug("OTEL: falha best-effort ao exportar %d spans async: %s", len(spans), exc)
        return 0
