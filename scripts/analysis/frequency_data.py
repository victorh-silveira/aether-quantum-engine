"""Coleta publica reproduzivel de OHLC M15/H1 e inventario historico."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from src.indicator.deriv import PublicDeriv


SYMBOLS = (
    "OTC_SPC",
    "OTC_NDX",
    "OTC_DJI",
    "OTC_GDAXI",
    "OTC_FTSE",
    "OTC_N225",
    "OTC_SX5E",
    "OTC_SSMI",
    "OTC_FCHI",
    "OTC_AEX",
    "OTC_HSI",
    "OTC_AS51",
)
START_EPOCH = int(datetime(2026, 1, 1, tzinfo=UTC).timestamp())
SCHEMA = {
    "symbol": pl.String,
    "period": pl.Int64,
    "epoch": pl.Int64,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
}


def file_hash(path: Path) -> str:
    """Identifica exatamente os bytes persistidos."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_inputs(output: Path) -> dict:
    """Congela configuracao, calendario e hashes sem sobrescrever snapshots anteriores."""
    output.mkdir(parents=True, exist_ok=True)
    for source_name, frozen_name in (
        ("config/settings.json", "settings_baseline.json"),
        ("data/economic_calendar_cache.json", "economic_calendar_snapshot.json"),
    ):
        source, frozen = Path(source_name), output / frozen_name
        if not frozen.is_file() and source.is_file():
            frozen.write_bytes(source.read_bytes())
    commit_path = output / "baseline_commit.txt"
    if not commit_path.is_file():
        head = Path(".git/HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            reference = Path(".git") / head[5:]
            head = reference.read_text(encoding="utf-8").strip() if reference.is_file() else head
        commit_path.write_text(head, encoding="utf-8")
    freeze_path = output / "freeze.json"
    if not freeze_path.is_file():
        sources = (
            "src/indicator/strategy_persistent_trend.py",
            "src/indicator/pullback.py",
            "src/domain/fundamental/economic_calendar.py",
            "src/domain/risk/pullback_stake.py",
        )
        freeze = {
            "git_revision": commit_path.read_text(encoding="utf-8").strip(),
            "frozen_at_utc": datetime.now(UTC).isoformat(),
            "source_sha256": {name: file_hash(Path(name)) for name in sources},
            "settings_baseline_sha256": file_hash(output / "settings_baseline.json"),
        }
        freeze_path.write_text(json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8")
    return json.loads(freeze_path.read_text(encoding="utf-8"))


def validate_rows(rows: list[dict], period: int, cutoff: int) -> tuple[list[dict], dict]:
    """Deduplica velas validas, rejeitando conflitos e registros ainda abertos."""
    valid: dict[int, dict] = {}
    conflicts: set[int] = set()
    invalid = duplicate = forming = 0
    for row in rows:
        try:
            epoch = int(row["epoch"])
            values = {name: float(row[name]) for name in ("open", "high", "low", "close")}
            if epoch % period or not all(math.isfinite(value) for value in values.values()):
                invalid += 1
                continue
            if values["high"] < max(values.values()) or values["low"] > min(values.values()):
                invalid += 1
                continue
        except (KeyError, TypeError, ValueError, OverflowError):
            invalid += 1
            continue
        if epoch + period > cutoff:
            forming += 1
            continue
        normalized = {"epoch": epoch, **values}
        if epoch in valid:
            duplicate += 1
            if valid[epoch] != normalized:
                conflicts.add(epoch)
        valid[epoch] = normalized
    clean = [valid[epoch] for epoch in sorted(valid) if epoch not in conflicts]
    gaps = [
        b["epoch"] - a["epoch"] for a, b in zip(clean, clean[1:], strict=False) if b["epoch"] - a["epoch"] != period
    ]
    inventory = {
        "rows": len(clean),
        "duplicates": duplicate,
        "conflicting_epochs": len(conflicts),
        "invalid": invalid,
        "forming_removed": forming,
        "first_epoch": clean[0]["epoch"] if clean else None,
        "last_epoch": clean[-1]["epoch"] if clean else None,
        "gap_count": len(gaps),
        "gap_seconds": sum(max(0, gap - period) for gap in gaps),
        "gap_note": "Lacunas incluem sessoes fechadas; nao imputar velas.",
    }
    return clean, inventory


async def collect_pair(
    source: Any,
    symbol: str,
    period: int,
    start: int,
    cutoff: int,
    *,
    delay: float = 2.0,
    seed_rows: list[dict] | None = None,
) -> tuple[list[dict], dict]:
    """Pagina para tras ate aquecimento fixo e detecta historico truncado."""
    warmup = 500
    fetch_start = start - warmup * period
    rows: list[dict] = list(seed_rows or [])
    cursor = min(int(row["epoch"]) for row in rows) - 1 if rows else cutoff - 1
    pages = 0
    error = ""
    rate_retries = 0
    out_of_range_page = None
    while pages < 200:
        valid_prefix, _ = validate_rows(rows, period, cutoff)
        if sum(row["epoch"] < start for row in valid_prefix) >= warmup:
            break
        try:
            page = await source.candles(symbol, count=5000, end=cursor, period=period)
        except Exception as exc:
            if "rate limit" in str(exc).lower() and rate_retries < 8:
                rate_retries += 1
                print(f"RATE_LIMIT {symbol} {period}: espera 30s tentativa {rate_retries}", flush=True)
                await asyncio.sleep(30)
                continue
            error = f"{type(exc).__name__}: {exc}"
            break
        pages += 1
        if not page:
            break
        oldest = min(int(row["epoch"]) for row in page)
        if oldest > cursor:
            error = "API retornou pagina posterior ao cursor"
            out_of_range_page = {
                "requested_end_epoch": cursor,
                "returned_first_epoch": oldest,
                "returned_last_epoch": max(int(row["epoch"]) for row in page),
            }
            break
        rows.extend(page)
        cursor = oldest - 1
        await asyncio.sleep(delay)
    clean, inventory = validate_rows(rows, period, cutoff)
    before = [row for row in clean if row["epoch"] < start]
    clean = before[-warmup:] + [row for row in clean if row["epoch"] >= start]
    inventory.update(
        {
            "symbol": symbol,
            "period": period,
            "pages": pages,
            "error": error,
            "requested_start_epoch": start,
            "requested_warmup_start_epoch": fetch_start,
            "warmup_bars_before_start": sum(row["epoch"] < start for row in clean),
            "rows_persisted": len(clean),
            "first_persisted_epoch": clean[0]["epoch"] if clean else None,
            "out_of_range_page": out_of_range_page,
        }
    )
    return [{"symbol": symbol, "period": period, **row} for row in clean], inventory


def calendar_inventory(path: Path, start: int, cutoff: int) -> dict:
    """Registra cache semanal sem afirmar cobertura anual a partir dos extremos."""
    if not path.is_file():
        return {"coverage_complete": False, "reason": "arquivo ausente", "event_count": 0}
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = payload.get("events", []) if isinstance(payload, dict) else payload
    epochs = []
    for event in events:
        try:
            timestamp = datetime.fromisoformat(str(event["date"]))
            if timestamp.tzinfo is not None:
                epochs.append(int(timestamp.timestamp()))
        except (KeyError, ValueError, TypeError):
            continue
    return {
        "path": str(path),
        "sha256": file_hash(path),
        "event_count": len(epochs),
        "first_epoch": min(epochs) if epochs else None,
        "last_epoch": max(epochs) if epochs else None,
        "coverage_complete": False,
        "reason": "Cache semanal nao certifica cobertura historica; exige arquivo arquivado verificavel.",
        "requested_start_epoch": start,
        "requested_end_epoch": cutoff,
    }


async def collect(output: Path, cutoff: int, start: int = START_EPOCH) -> dict:
    """Persiste cada par imediatamente e permite retomada pelo corte congelado."""
    output.mkdir(parents=True, exist_ok=True)
    freeze = freeze_inputs(output)
    calendar_path = output / "economic_calendar_snapshot.json"
    inventories = []
    pair_paths = []
    for symbol in SYMBOLS:
        for period in (900, 3600):
            target = output / f"{symbol}_{period}.parquet"
            meta_path = output / f"{symbol}_{period}.json"
            seed_rows = None
            prior_meta = None
            if target.is_file() and meta_path.is_file():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                warmup = 500
                if meta.get("cutoff_epoch") == cutoff and meta.get("start_epoch") == start:
                    if meta.get("sha256") != file_hash(target):
                        raise ValueError(f"Hash divergente: {target}")
                    if meta.get("warmup_bars_before_start", 0) >= warmup and not meta.get("error"):
                        inventories.append(meta)
                        pair_paths.append(target)
                        print(f"REUSO {symbol} {period}: {meta['rows_persisted']}", flush=True)
                        continue
                    seed_rows = pl.read_parquet(target).to_dicts()
                    prior_meta = meta
            source = PublicDeriv()
            try:
                rows, meta = await collect_pair(source, symbol, period, start, cutoff, seed_rows=seed_rows)
            finally:
                await source.close()
            if prior_meta is not None:
                for key in ("invalid", "duplicates", "conflicting_epochs", "forming_removed", "pages"):
                    meta[key] = meta.get(key, 0) + prior_meta.get(key, 0)
                meta["resumed_from_sha256"] = prior_meta["sha256"]
            frame = pl.DataFrame(rows, schema=SCHEMA)
            frame.write_parquet(target)
            meta.update({"cutoff_epoch": cutoff, "start_epoch": start, "sha256": file_hash(target)})
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            inventories.append(meta)
            pair_paths.append(target)
            print(f"COLETA {symbol} {period}: {len(rows)} erro={meta['error']}", flush=True)
    frame = pl.concat([pl.read_parquet(path) for path in pair_paths]).sort(["symbol", "period", "epoch"])
    target = output / "candles.parquet"
    frame.write_parquet(target)
    manifest = {
        "schema_version": 1,
        "source": "Deriv PublicDeriv ticks_history",
        "symbols": SYMBOLS,
        "start_epoch": start,
        "cutoff_epoch": cutoff,
        "cutoff_utc": datetime.fromtimestamp(cutoff, UTC).isoformat(),
        "year_partial": True,
        "november_partial": True,
        "dataset": str(target),
        "sha256": file_hash(target),
        "rows": len(frame),
        "pairs": inventories,
        "calendar": calendar_inventory(calendar_path, start, cutoff),
        "freeze": freeze,
        "qualifica_contrato": False,
        "qualifica_validacao_anual_completa": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    """Executa coleta com instante de corte congelado antes da primeira requisicao."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".cache/frequency-20261125"))
    parser.add_argument("--cutoff", type=int, default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cutoff_path = args.output / "cutoff_epoch.txt"
    if args.cutoff is not None:
        cutoff = args.cutoff
    elif cutoff_path.is_file():
        cutoff = int(cutoff_path.read_text().strip())
    else:
        cutoff = int(datetime.now(UTC).timestamp()) // 900 * 900
    cutoff_path.write_text(str(cutoff), encoding="utf-8")
    asyncio.run(collect(args.output, cutoff))


if __name__ == "__main__":
    main()
