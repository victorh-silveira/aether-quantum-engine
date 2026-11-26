"""Historico publico e exemplos causais separados cronologicamente."""

import asyncio
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from src.application.services.orchestrator.ws_bootstrap import resolve_public_ws_url
from src.domain.indicators.technical_indicators import compute_indicators
from src.domain.models.market_data import Candle
from src.infrastructure.inference.laya_context import serialize_laya_state
from src.infrastructure.inference.laya_decision_adapter import (
    _LAYA_QUESTIONS,
    build_laya_state,
    sanitize_settings_context,
)
from tools.laya_workflow.artifacts import file_digest, write_json


async def complete_history(path: Path, start: int, end: int, config: dict) -> list[Candle]:
    """Reutiliza parquet R_10 e completa paginas faltantes com market data publico."""
    from websockets.asyncio.client import connect  # noqa: PLC0415

    rows = pl.read_parquet(path).to_dicts() if path.exists() else []
    history = {int(row["epoch"]): row for row in rows if row.get("symbol") == "R_10" and row.get("granularity") == 300}
    expected = set(range(start, end, 300))
    missing = expected - history.keys()
    unavailable = set()
    if missing:
        async with connect(resolve_public_ws_url(config), open_timeout=30) as ws:
            cursor = max(missing)
            request_id = 0
            while missing:
                request_id += 1
                await ws.send(
                    json.dumps(
                        {
                            "ticks_history": "R_10",
                            "style": "candles",
                            "granularity": 300,
                            "end": cursor + 299,
                            "count": 5000,
                            "req_id": request_id,
                        }
                    )
                )
                response = json.loads(await asyncio.wait_for(ws.recv(), timeout=60))
                if "error" in response:
                    raise RuntimeError(f"Historico publico: {response['error']}")
                batch = response.get("candles", [])
                before = len(missing)
                for row in batch:
                    epoch = int(row["epoch"])
                    if epoch in expected:
                        history[epoch] = dict(row, symbol="R_10", granularity=300)
                missing = expected - history.keys()
                missing -= unavailable
                pl.DataFrame(list(history.values())).write_parquet(path)
                print(f"HISTORY | pagina={request_id} faltantes={len(missing)} cursor={cursor}", flush=True)
                if len(missing) == before:
                    if not batch or max(int(row["epoch"]) for row in batch) >= cursor:
                        raise ValueError("API nao disponibilizou historico anterior suficiente")
                    last = max(int(row["epoch"]) for row in batch)
                    gap = {epoch for epoch in missing if last < epoch <= cursor}
                    unavailable.update(gap)
                    missing -= gap
                    print(f"HISTORY_GAP | indisponiveis={len(gap)}; exemplos afetados serao excluidos", flush=True)
                cursor = max(missing) if missing else start
                await asyncio.sleep(0.25)
    selected = [history[epoch] for epoch in sorted(expected & history.keys())]
    bars = []
    for row in selected:
        epoch = int(row["epoch"])
        o, h, low, c = (float(row[key]) for key in ("open", "high", "low", "close"))
        if (
            epoch % 300
            or not all(math.isfinite(v) and v > 0 for v in (o, h, low, c))
            or not low <= min(o, c) <= max(o, c) <= h
        ):
            raise ValueError("Historico com OHLC ou epoch invalido")
        bars.append(Candle("R_10", o, h, low, c, datetime.fromtimestamp(epoch, tz=UTC), epoch))
    pl.DataFrame(selected).write_parquet(path)
    return bars


def build_splits(bars: list[Candle], config: dict, start: int, cutoff: int) -> tuple[dict, dict]:
    """Purga horizontes de fronteira e exclui empates sem misturar particoes."""
    span = cutoff - start
    edges = [start, *[start + int(span * frac // 300) * 300 for frac in (0.6, 0.7, 0.8)], cutoff]
    names = ("train", "calibration", "selection", "test")
    splits = {name: [] for name in names}
    stats = {"ties": 0, "purged": 0, "gaps": 0}
    for index in range(119, len(bars) - 1):
        current, future = bars[index : index + 2]
        observation = current.epoch + 300
        if not start <= observation < cutoff:
            continue
        window = bars[index - 119 : index + 1]
        if any(b.epoch - a.epoch != 300 for a, b in zip(window, [*window[1:], future], strict=True)):
            stats["gaps"] += 1
            continue
        part = next(i for i in range(4) if edges[i] <= observation < edges[i + 1])
        if future.epoch + 300 >= edges[part + 1]:
            stats["purged"] += 1
            continue
        if current.close == future.close:
            stats["ties"] += 1
            continue
        state = build_laya_state("R_10", window, compute_indicators(window).to_dict(), config)
        splits[names[part]].append(
            {
                "state": serialize_laya_state(state),
                "questions": _LAYA_QUESTIONS,
                "expected": {"direction": "CALL" if future.close > current.close else "PUT"},
                "observation_epoch": observation,
                "target_end": future.epoch + 300,
            }
        )
    return splits, dict(stats, edges=edges)


async def prepare(directory: Path, config: dict, cutoff: int, days: int = 90) -> dict:
    """Fixa corte e grava dados, contextos e manifesto sem carregar pesos."""
    if (directory / "manifest.json").exists():
        raise ValueError("Dataset ja fixado; use outro diretorio")
    directory.mkdir(parents=True, exist_ok=True)
    cutoff = cutoff // 300 * 300
    request_path = directory / "prepare_request.json"
    if request_path.exists():
        request = json.loads(request_path.read_text(encoding="utf-8"))
        cutoff, days = request["cutoff"], request["days"]
    else:
        write_json(request_path, {"cutoff": cutoff, "days": days})
    start = cutoff - days * 86400
    bars = await complete_history(directory / "candles.parquet", start - 120 * 300, cutoff, config)
    sanitized = sanitize_settings_context(config)
    splits, stats = build_splits(bars, sanitized, start, cutoff)
    for name, rows in splits.items():
        if not rows:
            raise ValueError(f"Particao vazia: {name}")
        (directory / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
    write_json(directory / "questions.json", _LAYA_QUESTIONS)
    write_json(directory / "settings.json", sanitized)
    files = ["candles.parquet", "questions.json", "settings.json", *[f"{name}.jsonl" for name in splits]]
    manifest = {
        "schema": 1,
        "context_version": 1,
        "symbol": "R_10",
        "period": 300,
        "cutoff": cutoff,
        "days": days,
        "source": "Deriv public ticks_history + parquet validado",
        "label": "proximo fechamento M5; aproximacao do contrato real",
        "stats": stats,
        "counts": {name: len(rows) for name, rows in splits.items()},
        "files": {name: file_digest(directory / name) for name in files},
    }
    write_json(directory / "manifest.json", manifest)
    return manifest
