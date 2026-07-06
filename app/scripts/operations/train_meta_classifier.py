from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any


os.environ["LOKY_MAX_CPU_COUNT"] = "4"


_APP = Path(__file__).resolve().parents[2]
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

import joblib
import numpy as np
import pandas as pd

from aether_paths import REPO_ROOT
from scripts.operations.train_meta_data import (
    META_TRAIN_DEFAULT_BARS,
    OhlcBundle,
    resolve_meta_train_bars,
    resolve_training_bundles,
)
from scripts.operations.train_meta_optuna import (
    build_paired_training_dataset,
    configure_meta_train_logging,
    run_optuna_study,
)


logger = logging.getLogger("META_TRAIN")
DEFAULT_DSN = "postgresql://aether:aether@localhost:5432/aether"
DEFAULT_OUTPUT = REPO_ROOT / "infra" / "docker" / "meta-models" / "meta_lgbm.pkl"
MIN_TARGET_VARIANCE = 1e-12


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


def validate_target_variance(y: np.ndarray, *, min_variance: float = MIN_TARGET_VARIANCE) -> None:
    variance = float(np.var(np.asarray(y, dtype=np.float64)))
    if variance <= float(min_variance):
        raise ValueError(
            "Dataset meta-classificador com variancia nula no alvo continuo detectado "
            f"(target_variance={variance}). "
            "Amplie o frame temporal com --bars 5000 ou busque historico em periodo de maior "
            "estresse de mercado antes de retreinar o meta-regressor."
        )


def target_variance(y: np.ndarray) -> float:
    return float(np.var(np.asarray(y, dtype=np.float64)))


def build_training_summary(
    *,
    frame: pd.DataFrame,
    y: np.ndarray,
    train_mae: float,
    val_mae: float,
    bundle_meta: dict[str, Any],
    output_path: Path,
    symbols: list[str],
    bundles: list[OhlcBundle],
) -> dict[str, Any]:
    return {
        "samples": int(len(frame)),
        "best_val_mae": float(val_mae),
        "train_mae": float(train_mae),
        "target_variance": target_variance(y),
        "output": str(output_path),
        "symbols": symbols,
        "granularity": int(bundles[0].granularity),
        "data_source": bundles[0].source,
        "feature_dim": int(bundle_meta["feature_dim"]),
        "model_type": str(bundle_meta.get("model_type", "regressor")),
        "bars_loaded": {b.symbol: int(len(b.closes)) for b in bundles},
        "cross_symbol_prob_delta_mean": float(frame["cross_symbol_prob_delta"].mean()),
    }


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
    frame, y, _proxy, _pnl = build_paired_training_dataset(
        bundles,
        micro_granularity=_micro_granularity(settings),
        fetch_count=resolve_meta_train_bars(bars),
    )
    validate_target_variance(y)
    model, bundle_meta, train_mae, val_mae = run_optuna_study(frame, y, trials=trials)
    _export_model(model, bundle_meta, output_path)
    summary = build_training_summary(
        frame=frame,
        y=y,
        train_mae=train_mae,
        val_mae=val_mae,
        bundle_meta=bundle_meta,
        output_path=output_path,
        symbols=symbols,
        bundles=bundles,
    )
    logger.info("Meta-regressor exportado: %s", summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina meta-regressor tabular com Optuna.")
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--granularity", type=int, default=60)
    parser.add_argument("--bars", type=int, default=META_TRAIN_DEFAULT_BARS)
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--symbols", nargs="+", default=["RDBULL", "RDBEAR"])
    parser.add_argument("--source", choices=("auto", "timescale", "deriv"), default="auto")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    configure_meta_train_logging()
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
