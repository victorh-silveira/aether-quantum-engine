"""Exposicao Prometheus do estado atual do indicador."""

from __future__ import annotations

import asyncio
import time

from indicator.domain import LiveObservation, Signal
from indicator.institutional import market_session_status_brt


class Metrics:
    """Gauge por ativo e periodo, limitado pelo catalogo ativo."""

    def __init__(self) -> None:
        self.signals: dict[tuple[str, int], Signal] = {}
        self.live: dict[tuple[str, int], LiveObservation] = {}
        self.symbols: dict[str, str] = {}
        self.reason_counts: dict[tuple[str, int, str], int] = {}
        self.deriv_rtt_ms: dict[str, float] = {}
        self.deriv_errors: dict[str, int] = {}
        self.last_tick_duration_seconds: float | None = None
        self._server: asyncio.Server | None = None

    def set_symbols(self, symbols: dict[str, str]) -> None:
        """Retira series de ativos inativos."""
        self.symbols = dict(symbols)
        self.signals = {key: value for key, value in self.signals.items() if key[0] in symbols}
        self.live = {key: value for key, value in self.live.items() if key[0] in symbols}

    def record_deriv_rtt(self, method: str, rtt_ms: float, *, ok: bool = True) -> None:
        """Registra latencia WS da Deriv e contagem de falhas por metodo."""
        key = str(method or "unknown")
        self.deriv_rtt_ms[key] = max(0.0, float(rtt_ms))
        if not ok:
            self.deriv_errors[key] = int(self.deriv_errors.get(key, 0)) + 1

    def record_tick_duration(self, duration_seconds: float) -> None:
        """Registra duracao do ultimo ciclo de observacao do indicador."""
        self.last_tick_duration_seconds = max(0.0, float(duration_seconds))

    def update(self, signal: Signal) -> None:
        """Guarda ultima leitura e contabiliza motivo por par/periodo."""
        self.signals[(signal.symbol, signal.period)] = signal
        key = (signal.symbol, signal.period, signal.reason)
        self.reason_counts[key] = int(self.reason_counts.get(key, 0)) + 1

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
        side_map = {"CALL": 1, "PUT": -1}
        reason_map = {"ok": 1, "setup_confirmado": 1, "experimental": 0}
        for (symbol, period), signal in sorted(self.signals.items()):
            labels = f'{{symbol="{symbol}",period="{period}"}}'
            lines.append(f"aether_indicator_side{labels} {side_map.get(signal.side, 0)}")
            if signal.probability is not None:
                lines.append(f"aether_indicator_probability{labels} {signal.probability:.8f}")
            lines.append(f"aether_indicator_close{labels} {signal.close:.8f}")
            lines.append(f"aether_indicator_candle_epoch{labels} {signal.candle_epoch}")
            lines.append(f"aether_indicator_fresh{labels} {int(now < signal.candle_epoch + 2 * period)}")
            lines.append(f"aether_indicator_validation{labels} {reason_map.get(signal.reason, -1)}")
            live = self.live.get((symbol, period))
            if (
                live is not None
                and signal.side in {"CALL", "PUT"}
                and live.signal_epoch == signal.candle_epoch
                and now - live.observed_at < 120
            ):
                lines.append(f"aether_indicator_live_spot{labels} {live.spot:.8f}")
                lines.append(f"aether_indicator_live_delta{labels} {live.delta:.8f}")
                lines.append(f"aether_indicator_live_side{labels} {side_map.get(live.interim_side, 0)}")
                lines.append(f"aether_indicator_live_observed_at{labels} {live.observed_at}")
        lines.extend(self._render_runtime_telemetry(now))
        return "\n".join(lines) + "\n"

    def _render_runtime_telemetry(self, now: float) -> list[str]:
        """Exporta bloqueios, latencia Deriv WS e abertura real vs Deriv OTC."""
        lines: list[str] = [
            "# TYPE aether_indicator_real_market_open gauge",
            "# TYPE aether_indicator_deriv_otc_open gauge",
        ]
        for symbol in sorted(self.symbols):
            real_open, otc_open = market_session_status_brt(symbol, now)
            lines.append(f'aether_indicator_real_market_open{{symbol="{symbol}"}} {int(real_open)}')
            lines.append(f'aether_indicator_deriv_otc_open{{symbol="{symbol}"}} {int(otc_open)}')
        if self.reason_counts:
            lines.append("# TYPE aether_indicator_reason_total counter")
            for (symbol, period, reason), count in sorted(self.reason_counts.items()):
                lines.append(
                    f'aether_indicator_reason_total{{symbol="{symbol}",period="{period}",reason="{reason}"}} {count}'
                )
        if self.deriv_rtt_ms:
            lines.append("# TYPE aether_indicator_deriv_ws_rtt_ms gauge")
            for method, rtt_ms in sorted(self.deriv_rtt_ms.items()):
                lines.append(f'aether_indicator_deriv_ws_rtt_ms{{method="{method}"}} {rtt_ms:.3f}')
        if self.deriv_errors:
            lines.append("# TYPE aether_indicator_deriv_ws_errors_total counter")
            for method, errs in sorted(self.deriv_errors.items()):
                lines.append(f'aether_indicator_deriv_ws_errors_total{{method="{method}"}} {errs}')
        if self.last_tick_duration_seconds is not None:
            lines.append("# TYPE aether_indicator_tick_duration_seconds gauge")
            lines.append(f"aether_indicator_tick_duration_seconds {self.last_tick_duration_seconds:.6f}")
        return lines

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Processa requisicoes HTTP do scrape de metricas."""
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

    async def start(self, host: str = "0.0.0.0", port: int = 9100) -> None:  # nosec B104 # noqa: S104
        """Abre endpoint local para scrape."""
        self._server = await asyncio.start_server(self._handle, host, port)

    async def close(self) -> None:
        """Fecha servidor."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
