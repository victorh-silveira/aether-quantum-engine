"""Exposicao Prometheus do estado atual do indicador."""

from __future__ import annotations

import asyncio
import time

from indicator.domain import LiveObservation, Signal


class Metrics:
    """Gauge por ativo e periodo, limitado pelo catalogo ativo."""

    def __init__(self) -> None:
        self.signals: dict[tuple[str, int], Signal] = {}
        self.live: dict[tuple[str, int], LiveObservation] = {}
        self.symbols: dict[str, str] = {}
        self._server: asyncio.Server | None = None

    def set_symbols(self, symbols: dict[str, str]) -> None:
        """Retira series de ativos inativos."""
        self.symbols = dict(symbols)
        self.signals = {key: value for key, value in self.signals.items() if key[0] in symbols}
        self.live = {key: value for key, value in self.live.items() if key[0] in symbols}

    def update(self, signal: Signal) -> None:
        """Guarda ultima leitura."""
        self.signals[(signal.symbol, signal.period)] = signal

    def update_live(self, observation: LiveObservation) -> None:
        """Guarda observação provisória sem modificar previsão confirmada."""
        self.live[(observation.symbol, observation.period)] = observation

    def render(self) -> str:
        """Gera metricas sem preencher lacunas historicas."""
        lines = [
            "# TYPE aether_indicator_catalog_active gauge",
            f"aether_indicator_catalog_active {len(self.symbols)}",
            "# TYPE aether_indicator_side gauge",
            "# TYPE aether_indicator_probability gauge",
            "# TYPE aether_indicator_close gauge",
            "# TYPE aether_indicator_candle_epoch gauge",
            "# TYPE aether_indicator_fresh gauge",
            "# TYPE aether_indicator_validation gauge",
            "# TYPE aether_indicator_live_spot gauge",
            "# TYPE aether_indicator_live_delta gauge",
            "# TYPE aether_indicator_live_side gauge",
            "# TYPE aether_indicator_live_observed_at gauge",
        ]
        now = time.time()
        for (symbol, period), signal in sorted(self.signals.items()):
            labels = f'{{symbol="{symbol}",period="{period}"}}'
            lines.append(f"aether_indicator_side{labels} {dict(CALL=1, PUT=-1).get(signal.side, 0)}")
            if signal.probability is not None:
                lines.append(f"aether_indicator_probability{labels} {signal.probability:.8f}")
            lines.append(f"aether_indicator_close{labels} {signal.close:.8f}")
            lines.append(f"aether_indicator_candle_epoch{labels} {signal.candle_epoch}")
            lines.append(f"aether_indicator_fresh{labels} {int(now < signal.candle_epoch + 2 * period)}")
            lines.append(
                f"aether_indicator_validation{labels} "
                f"{dict(ok=1, setup_confirmado=1, experimental=0).get(signal.reason, -1)}"
            )
            live = self.live.get((symbol, period))
            if (
                live is not None
                and signal.side in {"CALL", "PUT"}
                and live.signal_epoch == signal.candle_epoch
                and now - live.observed_at < 120
            ):
                lines.append(f"aether_indicator_live_spot{labels} {live.spot:.8f}")
                lines.append(f"aether_indicator_live_delta{labels} {live.delta:.8f}")
                lines.append(f"aether_indicator_live_side{labels} {dict(CALL=1, PUT=-1).get(live.interim_side, 0)}")
                lines.append(f"aether_indicator_live_observed_at{labels} {live.observed_at}")
        return "\n".join(lines) + "\n"

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request = await reader.readline()
            body = self.render().encode() if request.startswith(b"GET /metrics ") else b""
            status = b"200 OK" if body else b"404 Not Found"
            writer.write(
                b"HTTP/1.1 "
                + status
                + b"\r\nContent-Type: text/plain\r\nContent-Length: "
                + str(len(body)).encode()
                + b"\r\nConnection: close\r\n\r\n"
                + body
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    async def start(self, host: str = "0.0.0.0", port: int = 9100) -> None:
        """Abre endpoint local para scrape."""
        self._server = await asyncio.start_server(self._handle, host, port)

    async def close(self) -> None:
        """Fecha servidor."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
