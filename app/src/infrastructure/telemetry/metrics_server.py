"""Servidor HTTP assincrono embutido para exposicao de metricas ao Prometheus."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from src.infrastructure.telemetry.prometheus_exporter import PrometheusMetricsExporter


if TYPE_CHECKING:
    from src.application.services.quant_metrics_collector import QuantMetricsCollector

logger = logging.getLogger("AETH")


class MetricsServer:
    """Mini servidor assincrono para scrape de telemetria sem frameworks pesados."""

    def __init__(
        self,
        collector: QuantMetricsCollector,
        *,
        instrumentor: Any | None = None,
        host: str = "127.0.0.1",
        port: int = 9100,
    ) -> None:
        """Inicializa servidor com referencia ao coletor e instrumentador de negocio."""
        self._collector = collector
        self._instrumentor = instrumentor
        self._host = str(host)
        self._port = int(port)
        self._server: asyncio.Server | None = None

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Trata requisicao HTTP pura respondendo com payload OpenMetrics consolidado."""
        try:
            line = await reader.readline()
            req_line = line.decode("utf-8", errors="ignore").strip()

            if req_line.startswith("GET /metrics"):
                body = PrometheusMetricsExporter.format_metrics(self._collector)
                if self._instrumentor is not None and hasattr(self._instrumentor, "format_prometheus_metrics"):
                    body += self._instrumentor.format_prometheus_metrics()
                resp = (
                    "HTTP/1.1 200 OK\r\n"
                    "Content-Type: text/plain; version=0.0.4\r\n"
                    f"Content-Length: {len(body.encode('utf-8'))}\r\n"
                    "Connection: close\r\n\r\n"
                    f"{body}"
                )
            else:
                resp = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"

            writer.write(resp.encode("utf-8"))
            await writer.drain()
        except Exception as exc:
            logger.debug("METRICS_SERVER: Erro ao atender cliente: %s", exc)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception as exc:
                logger.debug("METRICS_SERVER: Erro ao fechar writer: %s", exc)

    async def start(self) -> None:
        """Inicia socket assincrono no loop do host."""
        self._server = await asyncio.start_server(self._handle_client, self._host, self._port)
        logger.info("METRICS_SERVER: Rodando em http://%s:%d/metrics", self._host, self._port)

    async def stop(self) -> None:
        """Encerra servidor liberando o socket."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
