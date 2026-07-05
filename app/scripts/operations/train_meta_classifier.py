from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any


_APP = Path(__file__).resolve().parents[2]
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

import joblib

from aether_paths import REPO_ROOT
from scripts.operations.train_meta_data import resolve_training_bundles
from scripts.operations.train_meta_optuna import build_paired_training_dataset, run_optuna_study


logger = logging.getLogger("META_TRAIN")
DEFAULT_DSN = "postgresql://aether:aether@localhost:5432/aether"
DEFAULT_OUTPUT = REPO_ROOT / "infra" / "docker" / "meta-models" / "meta_lgbm.pkl"


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


def _micro_granularity(settings: dict[str, Any]) -> int:
    data_cfg = settings.get("data_handler", {}) if isinstance(settings.get("data_handler"), dict) else {}
    return int(data_cfg.get("micro_granularity", 60)) if isinstance(data_cfg, dict) else 60


def _export_model(model, bundle_meta: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, **bundle_meta}, output_path)


async def train_meta_classifier(
    *,
    settings: dict[str, Any],
    dsn: str,
    symbols: list[str],
    granularity: int,
    bars: int,
    trials: int,
    output_path: Path,
    source: str,
) -> dict[str, Any]:
    bundles = await resolve_training_bundles(
        settings=settings,
        dsn=dsn,
        symbols=symbols,
        granularity=granularity,
        bars=bars,
        source=source,
    )
    frame, y, proxy, pnl = build_paired_training_dataset(
        bundles,
        micro_granularity=_micro_granularity(settings),
    )
    model, bundle_meta, score = run_optuna_study(frame, y, proxy, pnl, trials=trials)
    _export_model(model, bundle_meta, output_path)
    summary = {
        "samples": int(len(frame)),
        "best_edge_score": float(score),
        "output": str(output_path),
        "symbols": symbols,
        "granularity": int(bundles[0].granularity),
        "data_source": bundles[0].source,
        "feature_dim": int(bundle_meta["feature_dim"]),
        "bars_loaded": {b.symbol: int(len(b.closes)) for b in bundles},
    }
    logger.info("Meta-classificador exportado: %s", summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina meta-classificador tabular com Optuna.")
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--granularity", type=int, default=60)
    parser.add_argument("--bars", type=int, default=1024)
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--symbols", nargs="+", default=["RDBULL", "RDBEAR"])
    parser.add_argument("--source", choices=("auto", "timescale", "deriv"), default="auto")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    settings = _load_settings()
    dsn = _resolve_dsn(settings)
    summary = asyncio.run(
        train_meta_classifier(
            settings=settings,
            dsn=dsn,
            symbols=[str(s) for s in args.symbols],
            granularity=int(args.granularity),
            bars=int(args.bars),
            trials=int(args.trials),
            output_path=Path(args.output),
            source=str(args.source),
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
