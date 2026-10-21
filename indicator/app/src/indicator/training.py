"""Treino offline por ativo e periodo, com historico publico."""

from __future__ import annotations

import asyncio
import logging
import math
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

from indicator.deriv import PublicDeriv
from indicator.domain import PERIODS, aggregate, closed_candles, latest_contiguous, synthetic_symbols
from indicator.model import assess, load, save
from indicator.real_market_feed import fetch_real_candles

logger = logging.getLogger(__name__)
ASSET_TIMEOUT = 150
TRAIN_TIMEOUT = 3600
REQUEST_TIMEOUT = 40
MIN_CONTIGUOUS_BARS = 5000


async def history(source: PublicDeriv, symbol: str, now: int, pages: int = 6, period: int = 300) -> list:
    """Pagina velas do periodo sem incluir a vela corrente."""
    all_rows = []
    end: str | int = "latest"
    for page in range(1, pages + 1):
        rows = await asyncio.wait_for(source.candles(symbol, count=1000, end=end, period=period), REQUEST_TIMEOUT)
        logger.info("TRAIN %s %ds historico pagina %d/%d: %d velas", symbol, period, page, pages, len(rows))
        if not rows:
            break
        all_rows.extend(rows)
        oldest = min(int(row["epoch"]) for row in rows)
        end = oldest - 1
    candles = closed_candles(all_rows, now, period)
    gaps = [
        (left.epoch, right.epoch)
        for left, right in zip(candles, candles[1:], strict=False)
        if right.epoch - left.epoch > period
    ]
    for left, right in gaps[:3]:
        missing = (right - left) // period - 1
        if missing > 1000:
            continue
        rows = await asyncio.wait_for(
            source.candles(symbol, count=missing + 2, end=right - 1, period=period), REQUEST_TIMEOUT
        )
        logger.info("TRAIN %s %ds reparo %d..%d: %d velas", symbol, period, left, right, len(rows))
        all_rows.extend(rows)
    if symbol.startswith("OTC_"):
        try:
            real_data = await fetch_real_candles(symbol, period_seconds=period, range_str="1mo")
            for c in real_data:
                all_rows.append({"epoch": c.epoch, "open": c.open, "high": c.high, "low": c.low, "close": c.close})
            if real_data:
                logger.info("TRAIN %s %ds integrado com %d velas reais do mercado", symbol, period, len(real_data))
        except Exception as exc:
            logger.warning("Falha ao integrar velas reais para %s: %s", symbol, exc)
    return closed_candles(all_rows, now, period)


async def train_all(
    source: PublicDeriv,
    models: Path,
    now: int | None = None,
    retained_versions: dict[tuple[str, int], str] | None = None,
    candle_store=None,
) -> dict[str, str]:
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
            await asyncio.wait_for(
                _train_asset(source, models, timestamp, code, report, retained_versions or {}, candle_store),
                min(ASSET_TIMEOUT, remaining),
            )
        except (OSError, RuntimeError, ValueError, KeyError, TimeoutError) as exc:
            reason = str(exc) or "tempo limite por ativo"
            logger.warning("TRAIN %s indisponivel: %s", code, reason)
            for period in PERIODS:
                report.setdefault(f"{code}:{period}", f"SEM SINAL: {reason}")
        for period in PERIODS:
            logger.info("TRAIN resultado %s %ds: %s", code, period, report[f"{code}:{period}"])
    approved = sum(value.startswith("OK ") or value.startswith("VERSAO MANTIDA ") for value in report.values())
    experimental = sum(value.startswith("EXPERIMENTAL ") for value in report.values())
    logger.info(
        "TRAIN resumo: %d/%d pares aprovados; %d experimentais; %d sem sinal",
        approved,
        len(report),
        experimental,
        len(report) - approved - experimental,
    )
    return report


async def _train_asset(
    source: PublicDeriv,
    models: Path,
    timestamp: int,
    symbol: str,
    report: dict[str, str],
    retained_versions: dict[tuple[str, int], str],
    candle_store=None,
) -> None:
    """Treina periodos sem bloquear o loop durante calculo numerico."""
    base_candles = []
    for period in PERIODS:
        candles = latest_contiguous(await history(source, symbol, timestamp, period=period), period)
        if period == 300:
            base_candles = candles
        if candle_store is not None and candles:
            await candle_store.save_candles(symbol, period, candles)
        latest = datetime.fromtimestamp(candles[-1].epoch + period, UTC).isoformat() if candles else "nenhuma"
        logger.info("TRAIN %s %ds: %d velas continuas; ultima=%s", symbol, period, len(candles), latest)
        if len(candles) < MIN_CONTIGUOUS_BARS or (candles and candles[-1].epoch + 2 * period < timestamp):
            freshness = (
                "dados_atrasados"
                if candles and candles[-1].epoch + 2 * period < timestamp
                else "historico_continuo_insuficiente"
            )
            report[f"{symbol}:{period}"] = f"SEM SINAL: {freshness} ({len(candles)}/{MIN_CONTIGUOUS_BARS})"
            continue
        if period != 300:
            aggregated = {candle.epoch: candle for candle in aggregate(base_candles, period)}
            mismatches = sum(
                any(
                    not math.isclose(getattr(candle, field), getattr(aggregated[candle.epoch], field), rel_tol=1e-8)
                    for field in ("open", "high", "low", "close")
                )
                for candle in candles
                if candle.epoch in aggregated
            )
            if mismatches:
                report[f"{symbol}:{period}"] = f"SEM SINAL: divergencia OHLC M5 ({mismatches} velas)"
                continue
        result = await asyncio.to_thread(assess, symbol, period, candles)
        if result.training_steps:
            logger.info("TRAIN %s %ds: ajuste finalizado em %d/80 passos", symbol, period, result.training_steps)
        key = f"{symbol}:{period}"
        old = models / f"{symbol}_{period}.json"
        previous = load(models, symbol, period, allow_experimental=True)
        try:
            pilot_version = retained_versions.get((symbol, period))
            if (
                pilot_version
                and previous is not None
                and previous.status == "aprovado"
                and previous.version == pilot_version
            ):
                if result.model is not None:
                    save(result.model, models / "candidates" / result.model.version)
                    if candle_store is not None:
                        await candle_store.register_model(
                            symbol,
                            period,
                            result.model.version,
                            "piloto" if result.model.status == "aprovado" else "reprovado",
                        )
                report[key] = (
                    f"VERSAO MANTIDA {pilot_version} | candidato="
                    f"{result.model.version if result.model is not None else 'SEM SINAL'} "
                    f"acc={result.accuracy} brier={result.brier}"
                )
                continue
            if (
                old.is_file()
                and previous is not None
                and previous.status == "aprovado"
                and (result.model is None or result.model.status == "experimental")
            ):
                archive = models / "rejected"
                archive.mkdir(parents=True, exist_ok=True)
                destination = archive / f"{symbol}_{period}_{timestamp}_{time.time_ns()}.json"
                shutil.copy2(old, destination)
                logger.info("TRAIN %s %ds: checkpoint anterior arquivado em %s", symbol, period, destination)
                if result.model is None:
                    old.unlink()
            if result.model is None:
                report[key] = f"SEM SINAL: {result.reason}"
            else:
                save(result.model, models)
                if candle_store is not None:
                    await candle_store.register_model(
                        symbol,
                        period,
                        result.model.version,
                        "piloto" if result.model.status == "aprovado" else "reprovado",
                    )
                label = "OK" if result.model.status == "aprovado" else "EXPERIMENTAL"
                ece = f"{result.calibration_ece:.5f}" if result.calibration_ece is not None else "n/a"
                lower = f"{result.improvement_lower95:.5f}" if result.improvement_lower95 is not None else "n/a"
                report[key] = (
                    f"{label} {result.model.version} n_val={result.validation_samples} "
                    f"steps={result.training_steps}/80 "
                    f"acc={result.accuracy:.5f} ref={result.baseline_accuracy:.5f} "
                    f"brier={result.brier:.5f} ref={result.baseline_brier:.5f} "
                    f"ev_candidatas={result.ev_eligible_samples}/{result.validation_samples} "
                    f"p_lado_max={result.maximum_side_probability:.5f} "
                    f"ece={ece} lcb_brier={lower} "
                    f"features={result.model.feature_set} protocolo={result.model.validation_protocol} "
                    f"p10/p50/p90={result.probability_quantiles} regimes={result.regime_briers} "
                    f"reg={result.model.regularization:.3f} cal_slope={result.model.calibration_slope:.3f} "
                    f"candidatos_brier={result.candidate_briers} "
                    f"aptidao_ev={'COM_CANDIDATAS' if result.ev_eligible_samples else 'SEM_CANDIDATAS'}"
                )
        except OSError as exc:
            report[key] = f"SEM SINAL: falha ao gravar modelo: {exc}"
            logger.warning("TRAIN %s %ds: %s", symbol, period, report[key])
