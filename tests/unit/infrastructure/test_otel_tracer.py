"""Testes unitarios do rastreador OTel, metricas runtime Prometheus e spans de ciclo."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.infrastructure.telemetry.otel_business_instrumentor import BusinessMetricsInstrumentor
from src.infrastructure.telemetry.otel_tracer import (
    build_otlp_payload,
    current_span_id,
    current_trace_id,
    current_traceparent,
    drain_buffered_spans,
    flush_otlp_spans,
    flush_otlp_spans_async,
    trace_span,
)
from src.presentation.terminal.log_context import bind_log_context, clear_log_context, format_log_context_prefix


def test_trace_span_hierarchy_context_and_otlp_payload():
    drain_buffered_spans()
    assert current_trace_id() is None
    assert current_span_id() is None
    assert current_traceparent() is None

    with trace_span("root.cycle", attributes={"cycle.id": 1, "flag": True, "score": 0.75, "sym": "OTC_SSMI"}) as tid:
        assert current_trace_id() == tid
        sid_root = current_span_id()
        assert sid_root is not None
        assert current_traceparent() == f"00-{tid}-{sid_root}-01"
        bind_log_context(cycle_id=1, symbol="OTC_SSMI")
        try:
            assert format_log_context_prefix() == f"[c1|Swiss 20|t={tid[:8]}] "
        finally:
            clear_log_context()
        assert format_log_context_prefix() == f"[t={tid[:8]}] "
        with trace_span("child.order") as child_tid:
            assert child_tid == tid
            assert current_span_id() != sid_root
        with trace_span("isolated.cycle", new_root=True) as isolated_tid:
            assert isolated_tid != tid
            assert current_trace_id() == isolated_tid

    assert current_trace_id() is None
    with pytest.raises(ValueError, match="falha"), trace_span("failing.span"):
        raise ValueError("falha")

    spans = drain_buffered_spans()
    assert len(spans) == 4
    payload = build_otlp_payload(spans)
    assert len(payload["resourceSpans"]) == 1
    assert payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["parentSpanId"]


def test_flush_otlp_spans_sync_paths(monkeypatch):
    drain_buffered_spans()
    assert flush_otlp_spans("http://tempo:4318") == 0

    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    with trace_span("no.endpoint"):
        pass
    assert flush_otlp_spans() == 0

    ok_resp = MagicMock(status_code=200)
    with patch("src.infrastructure.telemetry.otel_tracer.httpx.post", return_value=ok_resp) as post_mock:
        with trace_span("ok.span"):
            pass
        assert flush_otlp_spans("http://tempo:4318/") == 1
        assert post_mock.call_args.args[0] == "http://tempo:4318/v1/traces"

    bad_resp = MagicMock(status_code=503)
    with patch("src.infrastructure.telemetry.otel_tracer.httpx.post", return_value=bad_resp):
        with trace_span("bad.http"):
            pass
        assert flush_otlp_spans("http://tempo:4318/v1/traces") == 0

    with patch("src.infrastructure.telemetry.otel_tracer.httpx.post", side_effect=OSError("conn")):
        with trace_span("err.http"):
            pass
        assert flush_otlp_spans("http://tempo:4318") == 0


@pytest.mark.asyncio
async def test_flush_otlp_spans_async_paths(monkeypatch):
    drain_buffered_spans()
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    with trace_span("async.no_url"):
        pass
    assert await flush_otlp_spans_async() == 0
    assert await flush_otlp_spans_async("http://tempo:4318") == 0

    client_ok = AsyncMock()
    client_ok.__aenter__ = AsyncMock(return_value=client_ok)
    client_ok.__aexit__ = AsyncMock(return_value=False)
    client_ok.post = AsyncMock(return_value=MagicMock(status_code=200))
    with patch("src.infrastructure.telemetry.otel_tracer.httpx.AsyncClient", return_value=client_ok):
        with trace_span("async.ok"):
            pass
        assert await flush_otlp_spans_async("http://tempo:4318") == 1

    client_bad = AsyncMock()
    client_bad.__aenter__ = AsyncMock(return_value=client_bad)
    client_bad.__aexit__ = AsyncMock(return_value=False)
    client_bad.post = AsyncMock(return_value=MagicMock(status_code=500))
    with patch("src.infrastructure.telemetry.otel_tracer.httpx.AsyncClient", return_value=client_bad):
        with trace_span("async.bad"):
            pass
        assert await flush_otlp_spans_async("http://tempo:4318") == 0

    client_err = AsyncMock()
    client_err.__aenter__ = AsyncMock(return_value=client_err)
    client_err.__aexit__ = AsyncMock(return_value=False)
    client_err.post = AsyncMock(side_effect=RuntimeError("timeout"))
    with patch("src.infrastructure.telemetry.otel_tracer.httpx.AsyncClient", return_value=client_err):
        with trace_span("async.err"):
            pass
        assert await flush_otlp_spans_async("http://tempo:4318") == 0


def test_runtime_telemetry_metrics_export():
    inst = BusinessMetricsInstrumentor()
    inst.runtime.stop_win_reached = True
    inst.runtime.last_history_sync_seconds = 12.5
    inst.runtime.record_cycle_duration(0.42)
    inst.runtime.record_cycle_skip("cooldown")
    inst.runtime.record_deriv_rtt("buy", 85.0, ok=True)
    inst.runtime.record_deriv_rtt("proposal", 120.0, ok=False)
    inst.runtime.record_quote_payout("OTC_SSMI", 10.0, 17.2)
    inst.runtime.record_quote_payout("OTC_BAD", 0.0, 10.0)
    text = inst.format_prometheus_metrics()
    assert "aether_session_stop_win_reached 1" in text
    assert "aether_cycle_duration_seconds 0.420000" in text
    assert "aether_history_sync_duration_seconds 12.500" in text
    assert 'aether_cycle_skips_total{reason="cooldown"} 1' in text
    assert 'aether_deriv_ws_rtt_ms{method="buy"} 85.000' in text
    assert 'aether_deriv_ws_errors_total{method="proposal"} 1' in text
    assert 'aether_quote_observed_payout_rate{symbol="OTC_SSMI"} 0.720000' in text
    assert 'aether_quote_observed_breakeven_prob{symbol="OTC_SSMI"} 0.581395' in text
