"""Treino offline por ativo e periodo, com historico publico."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from indicator.deriv import PublicDeriv
from indicator.domain import PERIODS, aggregate, closed_candles, latest_contiguous, synthetic_symbols
from indicator.model import save, train

logger = logging.getLogger(__name__)


async def history(source: PublicDeriv, symbol: str, now: int, pages: int = 6) -> list:
    """Pagina velas M5 sem incluir a vela corrente."""
    all_rows = []
    end: str | int = "latest"
    for _ in range(pages):
        rows = await source.candles(symbol, count=1000, end=end)
        if not rows:
            break
        all_rows.extend(rows)
        oldest = min(int(row["epoch"]) for row in rows)
        end = oldest - 1
    return closed_candles(all_rows, now)


async def train_all(source: PublicDeriv, models: Path, now: int | None = None) -> dict[str, str]:
    """Treina cada ativo e periodo e reporta pares sem evidencia validada."""
    timestamp = int(time.time()) if now is None else now
    symbols = synthetic_symbols(await source.symbols())
    if not symbols:
        raise RuntimeError("catalogo sintetico vazio")
    report = {}
    for symbol in symbols:
        try:
            base = await history(source, symbol, timestamp)
            for period in PERIODS:
                candles = latest_contiguous(aggregate(base, period), period)
                model = train(symbol, period, candles)
                key = f"{symbol}:{period}"
                if model is None:
                    report[key] = "SEM SINAL: sem amostras ou sem ganho temporal"
                else:
                    save(model, models)
                    report[key] = (
                        f"OK {model.version} acc={model.validation_accuracy:.3f} brier={model.validation_brier:.3f}"
                    )
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            logger.warning("TRAIN %s indisponivel: %s", symbol, exc)
            for period in PERIODS:
                report[f"{symbol}:{period}"] = f"SEM SINAL: {exc}"
    return report
