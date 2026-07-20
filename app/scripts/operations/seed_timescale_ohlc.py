from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any


_APP = Path(__file__).resolve().parents[2]
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

from aether_asyncio import run_async, silence_asyncio_debug
from aether_paths import REPO_ROOT
from scripts.operations.train_meta_data import (
    META_TRAIN_DEFAULT_BARS,
    load_bundles_from_deriv,
    persist_bundles_to_timescale,
    resolve_meta_train_bars,
)


logger = logging.getLogger("META_TRAIN")
DEFAULT_DSN = "postgresql://aether:aether@localhost:5432/aether"


def _load_settings() -> dict[str, Any]:
    settings_path = REPO_ROOT / "config" / "settings.json"
    if settings_path.is_file():
        return json.loads(settings_path.read_text(encoding="utf-8"))
    return {}


def _resolve_dsn(settings: dict[str, Any]) -> str:
    infra = settings.get("infra", {})
    chunk = infra.get("timescale", {}) if isinstance(infra, dict) else {}
    if isinstance(chunk, dict) and chunk.get("dsn"):
        return str(chunk["dsn"])
    return os.getenv("AETHER_TIMESCALE_DSN", DEFAULT_DSN)


def _default_granularities(settings: dict[str, Any]) -> list[int]:
    data = settings.get("data_handler") if isinstance(settings.get("data_handler"), dict) else {}
    micro = int(data.get("micro_granularity", 120)) if isinstance(data, dict) else 120
    macro = int(data.get("granularity", 600)) if isinstance(data, dict) else 600
    ordered = [micro, macro]
    unique: list[int] = []
    for value in ordered:
        if value not in unique:
            unique.append(value)
    return unique


async def seed_timescale_ohlc(
    *,
    settings: dict[str, Any],
    dsn: str,
    symbols: list[str],
    granularities: list[int],
    bars: int,
) -> dict[str, Any]:
    bar_target = resolve_meta_train_bars(bars)
    summary: dict[str, Any] = {"symbols": symbols, "granularities": {}, "bars_target": bar_target}
    for gran in granularities:
        bundles = await load_bundles_from_deriv(settings, symbols, int(gran), bar_target)
        written = await persist_bundles_to_timescale(dsn, bundles)
        summary["granularities"][str(int(gran))] = {
            "bars_loaded": {b.symbol: int(len(b.closes)) for b in bundles},
            "rows_written": int(written),
        }
        logger.info(
            "SEED_TIMESCALE | gran=%ds | symbols=%d | rows_written=%d",
            int(gran),
            len(bundles),
            written,
        )
    return summary


def _parse_args(settings: dict[str, Any]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Popula Timescale OHLC (micro+macro) via Deriv.")
    parser.add_argument("--bars", type=int, default=META_TRAIN_DEFAULT_BARS)
    parser.add_argument("--symbols", nargs="+", default=["R_10"])
    parser.add_argument(
        "--granularity",
        type=int,
        nargs="+",
        default=_default_granularities(settings),
    )
    return parser.parse_args()


def main() -> None:
    silence_asyncio_debug()
    logging.basicConfig(level=logging.INFO)
    settings = _load_settings()
    args = _parse_args(settings)
    summary = run_async(
        seed_timescale_ohlc(
            settings=settings,
            dsn=_resolve_dsn(settings),
            symbols=[str(s) for s in args.symbols],
            granularities=[int(g) for g in args.granularity],
            bars=int(args.bars),
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
