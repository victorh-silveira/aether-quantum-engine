"""Metricas numericas e endpoint Prometheus."""

import asyncio

import pytest

from indicator.domain import LiveObservation, Signal
from indicator.metrics import Metrics


def test_metrics_emit_only_observed_values(monkeypatch):
    monkeypatch.setattr("indicator.metrics.time.time", lambda: 1270)
    metrics = Metrics()
    metrics.set_symbols({"R_75": "V75"})
    metrics.update(Signal("R_75", 300, 300, "SEM SINAL", None, None, "modelo_indisponivel", 100))
    text = metrics.render()
    assert 'aether_indicator_side{symbol="R_75",period="300"} 0' in text
    assert 'aether_indicator_validation{symbol="R_75",period="300"} -1' in text
    assert "aether_indicator_probability{" not in text
    metrics.update(Signal("R_75", 300, 600, "PUT", 0.4, "v1", "ok", 99))
    assert 'aether_indicator_side{symbol="R_75",period="300"} -1' in metrics.render()
    assert 'aether_indicator_validation{symbol="R_75",period="300"} 1' in metrics.render()
    metrics.update(Signal("R_75", 300, 900, "CALL", 0.6, "exp", "experimental", 101))
    assert 'aether_indicator_validation{symbol="R_75",period="300"} 0' in metrics.render()
    metrics.update_live(LiveObservation("R_75", 300, 900, 1260, 102, 1, "CALL"))
    assert 'aether_indicator_live_spot{symbol="R_75",period="300"} 102.00000000' in metrics.render()
    assert 'aether_indicator_live_side{symbol="R_75",period="300"} 1' in metrics.render()
    metrics.update(Signal("R_75", 300, 900, "SEM SINAL", None, None, "modelo_atualizado", 101))
    assert 'aether_indicator_live_spot{symbol="R_75",period="300"}' not in metrics.render()
    metrics.update(Signal("R_75", 300, 900, "CALL", 0.6, "exp", "experimental", 101))
    monkeypatch.setattr("indicator.metrics.time.time", lambda: 1400)
    assert 'aether_indicator_live_spot{symbol="R_75",period="300"}' not in metrics.render()
    metrics.update(Signal("R_75", 300, 1200, "PUT", 0.4, "exp", "experimental", 102))
    assert 'aether_indicator_live_spot{symbol="R_75",period="300"}' not in metrics.render()
    metrics.set_symbols({})
    assert not metrics.signals
    assert not metrics.live


@pytest.mark.asyncio
async def test_metrics_http_endpoint():
    metrics = Metrics()
    await metrics.start(host="127.0.0.1", port=0)
    port = metrics._server.sockets[0].getsockname()[1]
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(b"GET /metrics HTTP/1.1\r\n\r\n")
    await writer.drain()
    response = await reader.read()
    assert b"200 OK" in response and b"aether_indicator_catalog_active" in response
    writer.close()
    await writer.wait_closed()
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(b"GET /bad HTTP/1.1\r\n\r\n")
    await writer.drain()
    assert b"404 Not Found" in await reader.read()
    writer.close()
    await writer.wait_closed()
    await metrics.close()
    await metrics.close()
