"""Coleta publica de ticks e propostas Touch/No Touch; nao possui caminho de compra."""

import argparse
import asyncio
import logging
import sys
from time import monotonic
from pathlib import Path
from types import SimpleNamespace


_APP = Path(__file__).resolve().parents[2]
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

from scripts.operations.collect_public_ticks import parse_tick
from src.application.services.orchestrator.ws_bootstrap import resolve_public_ws_url
from src.domain.config_knobs import load_settings_json
from src.domain.models.touch_policy import resolve_touch_policy
from src.infrastructure.api.websocket_manager import WebSocketManager
from src.infrastructure.handlers.tick_buffer import TickBuffer
from src.infrastructure.handlers.touch_broker import quote_touch_candidates
from src.infrastructure.market.timescale_writer import TimescaleMarketWriter
from src.infrastructure.market.touch_store import append_touch_quotes


async def collect_dataset(settings: dict, *, max_groups: int = 0) -> int:
    """Acumula observacoes completas; queda WSS encerra explicitamente para nao ocultar lacunas."""
    policy = resolve_touch_policy(settings)
    logger = logging.getLogger("AETH.touch.collect")
    ws = WebSocketManager(resolve_public_ws_url(settings))
    writer = TimescaleMarketWriter(dsn=settings["infra"]["timescale"]["dsn"])
    buffer = TickBuffer([policy.symbol])
    groups, next_quote = 0, monotonic() + policy.history_seconds

    async def on_tick(data):
        tick = parse_tick(data)
        if tick is not None:
            buffer.record_tick(policy.symbol, *tick)
            await writer.enqueue_tick(symbol=policy.symbol, epoch_ms=tick[0], price=tick[1])

    try:
        if not await writer.ping():
            raise ConnectionError("Timescale indisponivel")
        await ws.connect()
        ws.subscriptions["tick"] = on_tick
        response = await ws.send({"ticks": policy.symbol, "subscribe": 1})
        if response.get("error"):
            raise RuntimeError("Subscricao de ticks rejeitada")
        handler = SimpleNamespace(ws=ws, logger=logger)
        while max_groups <= 0 or groups < max_groups:
            if not ws.is_running:
                raise ConnectionError("WSS interrompido; reinicie coleta, janelas incompletas serao rejeitadas")
            if monotonic() >= next_quote:
                next_quote = monotonic() + policy.duration_seconds
                try:
                    quotes = await quote_touch_candidates(handler, buffer.recent_ticks(policy.symbol), 1.0, policy)
                    await asyncio.to_thread(append_touch_quotes, quotes)
                    groups += int(bool(quotes))
                    logger.info("TOUCH DATA | grupos=%d propostas=%d", groups, len(quotes))
                except (ValueError, KeyError, IndexError) as exc:
                    logger.warning("TOUCH DATA | janela rejeitada: %s", exc)
            await asyncio.sleep(1)
        await asyncio.sleep(policy.duration_seconds + max(policy.latency_ms) / 1000 + 2)
        return groups
    finally:
        await ws.close()
        await writer.close()


def main() -> int:
    """CLI sem credenciais e sem endpoints buy/bulk_purchase."""
    parser = argparse.ArgumentParser(description="Coleta publica Touch/No Touch para treino")
    parser.add_argument("--max-groups", type=int, default=0, help="0=continuo; Ctrl+C encerra")
    args = parser.parse_args()
    asyncio.run(collect_dataset(load_settings_json(), max_groups=args.max_groups))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
