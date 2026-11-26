"""Verifica contexto completo e respostas do checkpoint local sem enviar ordens."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from datetime import UTC, datetime
from pathlib import Path

from src.domain.indicators.technical_indicators import compute_indicators
from src.domain.models.market_data import Candle
from src.infrastructure.inference.laya_decision_adapter import LayaDecisionAdapter


async def main() -> None:
    """Compara cenarios sinteticos usando exclusivamente pesos ja armazenados."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = json.loads(Path("config/settings.json").read_text(encoding="utf-8"))
    adapter = LayaDecisionAdapter()
    for label, slope in (("alta", 1.5), ("baixa", -1.5), ("lateral", 0.0)):
        bars = []
        for index in range(120):
            epoch = 1_764_000_000 + index * 300
            opening = 4800 + index * slope + math.sin(index * 0.7) * 2
            close = opening + slope + math.sin(index) * 0.5
            bars.append(
                Candle(
                    "R_10",
                    opening,
                    max(opening, close) + 1,
                    min(opening, close) - 1,
                    close,
                    datetime.fromtimestamp(epoch, tz=UTC),
                    epoch,
                )
            )
        decision = await adapter.decide("R_10", bars, compute_indicators(bars).to_dict(), settings)
        print(
            json.dumps(
                {
                    "cenario": label,
                    "dir": decision.direction.name,
                    "p_call": decision.prob_call,
                    "p_put": decision.prob_put,
                    "confidence": decision.confidence,
                    "latency_ms": decision.latency_ms,
                }
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
