"""Testes unitarios para o mini-servidor assincrono de metricas."""

import asyncio

import pytest

from src.application.services.quant_metrics_collector import QuantMetricsCollector
from src.infrastructure.telemetry.metrics_server import MetricsServer
from src.infrastructure.telemetry.otel_business_instrumentor import BusinessMetricsInstrumentor


@pytest.mark.asyncio
async def test_metrics_server_start_request_and_stop():
    collector = QuantMetricsCollector(initial_balance=5000.0)
    collector.record_cycle(is_execution=True)

    server = MetricsServer(collector, host="127.0.0.1", port=9199)
    await server.start()

    reader, writer = await asyncio.open_connection("127.0.0.1", 9199)
    writer.write(b"GET /metrics HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
    await writer.drain()

    data = await reader.read(4096)
    writer.close()
    await writer.wait_closed()
    await server.stop()

    resp = data.decode("utf-8")
    assert "HTTP/1.1 200 OK" in resp
    assert "aether_current_balance_usd 5000.0" in resp


@pytest.mark.asyncio
async def test_metrics_server_with_business_instrumentor():
    collector = QuantMetricsCollector(initial_balance=8000.0)
    instrumentor = BusinessMetricsInstrumentor()
    instrumentor.update_balance(8000.0)
    instrumentor.record_trade("1HZ75V", "PUT", is_win=True, profit_usd=85.0, predicted_prob=0.7)

    server = MetricsServer(collector, instrumentor=instrumentor, host="127.0.0.1", port=9197)
    await server.start()

    reader, writer = await asyncio.open_connection("127.0.0.1", 9197)
    writer.write(b"GET /metrics HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
    await writer.drain()

    data = await reader.read(4096)
    writer.close()
    await writer.wait_closed()
    await server.stop()

    resp = data.decode("utf-8")
    assert "HTTP/1.1 200 OK" in resp
    assert "aether_trading_balance_usd 8085.0" in resp
    assert 'aether_trading_contracts_total{symbol="1HZ75V",direction="PUT",outcome="WIN"} 1' in resp


@pytest.mark.asyncio
async def test_metrics_server_not_found():
    collector = QuantMetricsCollector()
    server = MetricsServer(collector, host="127.0.0.1", port=9198)
    await server.start()

    reader, writer = await asyncio.open_connection("127.0.0.1", 9198)
    writer.write(b"GET /unknown HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
    await writer.drain()

    data = await reader.read(1024)
    writer.close()
    await writer.wait_closed()
    await server.stop()

    assert "HTTP/1.1 404 Not Found" in data.decode("utf-8")


@pytest.mark.asyncio
async def test_metrics_server_handle_client_exception():
    collector = QuantMetricsCollector()
    server = MetricsServer(collector, host="127.0.0.1", port=9196)

    class BrokenReader:
        async def readline(self):
            raise ConnectionResetError("Conexao abortada")

    class BrokenWriter:
        def close(self):
            pass

        async def wait_closed(self):
            raise OSError("Falha ao fechar socket")

    await server._handle_client(BrokenReader(), BrokenWriter())


@pytest.mark.asyncio
async def test_metrics_server_start_bind_oserror():
    collector = QuantMetricsCollector()
    server1 = MetricsServer(collector, host="127.0.0.1", port=9195)
    await server1.start()

    server2 = MetricsServer(collector, host="127.0.0.1", port=9195)
    await server2.start()
    assert server2._server is None

    await server2.stop()
    await server1.stop()
