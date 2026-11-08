"""Captura ticks publicos do 1HZ75V sem autenticar nem abrir trades."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import sys
from pathlib import Path

import websockets


_APP = Path(__file__).resolve().parents[2]
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

from src.application.services.orchestrator.ws_bootstrap import resolve_public_ws_url
from src.domain.config_knobs import load_settings_json
from src.infrastructure.api.websocket_connect import connect_wss_with_ip_failover
from src.infrastructure.market.timescale_writer import TimescaleMarketWriter


logger = logging.getLogger("AETH.ticks")
SYMBOL = "1HZ75V"


def parse_tick(data: dict) -> tuple[int, float] | None:
    """Aceita apenas quote finita do simbolo fixo e epoch positivo."""
    tick = data.get("tick")
    if not isinstance(tick, dict) or tick.get("symbol") != SYMBOL:
        return None
    try:
        epoch_ms = int(float(tick["epoch"]) * 1000)
        price = float(tick["quote"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if epoch_ms <= 0 or not math.isfinite(price) or price <= 0:
        return None
    return epoch_ms, price


async def collect_ticks(
    writer: TimescaleMarketWriter,
    url: str,
    *,
    max_ticks: int = 0,
    max_retries: int = 0,
) -> int:
    """Assina WSS publico e reconecta; max_ticks=0 significa execucao continua."""
    count = 0
    failures = 0
    while max_ticks <= 0 or count < max_ticks:
        socket = None
        try:
            socket = await connect_wss_with_ip_failover(url, open_timeout=20.0)
            await socket.send(json.dumps({"ticks": SYMBOL, "subscribe": 1}))
            async for message in socket:
                data = json.loads(message)
                if data.get("error"):
                    raise RuntimeError(f"Deriv tick subscription: {data['error'].get('message', 'erro')}")
                tick = parse_tick(data)
                if tick is None:
                    continue
                await writer.enqueue_tick(symbol=SYMBOL, epoch_ms=tick[0], price=tick[1])
                count += 1
                failures = 0
                if count % 1000 == 0:
                    logger.info("TICKS | %s | capturados=%d", SYMBOL, count)
                if max_ticks > 0 and count >= max_ticks:
                    break
        except (OSError, ConnectionError, TimeoutError, websockets.WebSocketException) as exc:
            failures += 1
            logger.warning("TICKS | WSS interrompido tentativa=%d erro=%s", failures, exc)
            if max_retries > 0 and failures >= max_retries:
                raise
            await asyncio.sleep(min(30.0, float(2 ** min(failures, 5))))
        finally:
            if socket is not None:
                await socket.close()
    await writer.flush()
    return count


async def main() -> int:
    """Valida Timescale e executa captura sem qualquer endpoint de trading."""
    parser = argparse.ArgumentParser(description="Captura ticks publicos de 1HZ75V no TimescaleDB")
    parser.add_argument("--max-ticks", type=int, default=0, help="0=continuo; valor positivo para smoke test")
    parser.add_argument("--max-retries", type=int, default=0, help="0=ilimitado")
    args = parser.parse_args()
    settings = load_settings_json()
    ts = settings["infra"]["timescale"]
    writer = TimescaleMarketWriter(dsn=str(ts["dsn"]))
    try:
        if not await writer.ping():
            raise ConnectionError("TimescaleDB indisponivel")
        count = await collect_ticks(
            writer,
            resolve_public_ws_url(settings),
            max_ticks=args.max_ticks,
            max_retries=args.max_retries,
        )
        logger.info("TICKS | %s | concluido count=%d", SYMBOL, count)
        return 0
    finally:
        await writer.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(asyncio.run(main()))
