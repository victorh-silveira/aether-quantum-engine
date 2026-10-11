"""Treino offline por ativo e periodo, com historico publico."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

from indicator.deriv import PublicDeriv
from indicator.domain import PERIODS, aggregate, closed_candles, latest_contiguous, synthetic_symbols
from indicator.model import assess, load, save

logger = logging.getLogger(__name__)
ASSET_TIMEOUT = 90
TRAIN_TIMEOUT = 900
REQUEST_TIMEOUT = 40


async def history(source: PublicDeriv, symbol: str, now: int, pages: int = 12) -> list:
    """Pagina velas M5 sem incluir a vela corrente."""
    all_rows = []
    end: str | int = "latest"
    for page in range(1, pages + 1):
        rows = await asyncio.wait_for(source.candles(symbol, count=1000, end=end), REQUEST_TIMEOUT)
        logger.info("TRAIN %s historico pagina %d/%d: %d velas", symbol, page, pages, len(rows))
        if not rows:
            break
        all_rows.extend(rows)
        oldest = min(int(row["epoch"]) for row in rows)
        end = oldest - 1
    return closed_candles(all_rows, now)


async def train_all(source: PublicDeriv, models: Path, now: int | None = None) -> dict[str, str]:
    """Treina cada ativo e periodo e reporta pares sem evidencia validada."""
    timestamp = int(time.time()) if now is None else now
    symbols = synthetic_symbols(await asyncio.wait_for(source.symbols(), REQUEST_TIMEOUT))
    if not symbols:
        raise RuntimeError("catalogo sintetico vazio")
    report = {}
    deadline = time.monotonic() + TRAIN_TIMEOUT
    logger.info("TRAIN iniciado: %d indices Volatility, periodos M5/M15/H1", len(symbols))
    for index, symbol in enumerate(symbols.items(), 1):
        code, name = symbol
        logger.info("TRAIN ativo %d/%d: %s (%s)", index, len(symbols), code, name)
        try:
            remaining = max(0, deadline - time.monotonic())
            if not remaining:
                raise TimeoutError("tempo total do treino esgotado")
            await asyncio.wait_for(_train_asset(source, models, timestamp, code, report), min(ASSET_TIMEOUT, remaining))
        except (OSError, RuntimeError, ValueError, KeyError, TimeoutError) as exc:
            reason = str(exc) or "tempo limite por ativo"
            logger.warning("TRAIN %s indisponivel: %s", code, reason)
            for period in PERIODS:
                report.setdefault(f"{code}:{period}", f"SEM SINAL: {reason}")
        for period in PERIODS:
            logger.info("TRAIN resultado %s %ds: %s", code, period, report[f"{code}:{period}"])
    approved = sum(value.startswith("OK ") for value in report.values())
    experimental = sum(value.startswith("EXPERIMENTAL ") for value in report.values())
    logger.info(
        "TRAIN resumo: %d/%d pares aprovados; %d experimentais; %d sem sinal",
        approved,
        len(report),
        experimental,
        len(report) - approved - experimental,
    )
    return report


async def _train_asset(source: PublicDeriv, models: Path, timestamp: int, symbol: str, report: dict[str, str]) -> None:
    """Treina periodos sem bloquear o loop durante calculo numerico."""
    base = await history(source, symbol, timestamp)
    for period in PERIODS:
        candles = latest_contiguous(aggregate(base, period), period)
        latest = datetime.fromtimestamp(candles[-1].epoch + period, UTC).isoformat() if candles else "nenhuma"
        logger.info("TRAIN %s %ds: %d velas continuas; ultima=%s", symbol, period, len(candles), latest)
        result = await asyncio.to_thread(assess, symbol, period, candles)
        key = f"{symbol}:{period}"
        old = models / f"{symbol}_{period}.json"
        previous = load(models, symbol, period, allow_experimental=True)
        if (
            old.is_file()
            and previous is not None
            and previous.status == "aprovado"
            and (result.model is None or result.model.status == "experimental")
        ):
            archive = models / "rejected"
            archive.mkdir(parents=True, exist_ok=True)
            destination = archive / f"{symbol}_{period}_{timestamp}_{time.time_ns()}.json"
            old.replace(destination)
            logger.info("TRAIN %s %ds: checkpoint anterior arquivado em %s", symbol, period, destination)
        if result.model is None:
            report[key] = f"SEM SINAL: {result.reason}"
        else:
            save(result.model, models)
            label = "OK" if result.model.status == "aprovado" else "EXPERIMENTAL"
            report[key] = (
                f"{label} {result.model.version} n_val={result.validation_samples} "
                f"acc={result.accuracy:.5f} ref={result.baseline_accuracy:.5f} "
                f"brier={result.brier:.5f} ref={result.baseline_brier:.5f}"
            )
