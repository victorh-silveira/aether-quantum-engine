from __future__ import annotations

import argparse
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

from aether_asyncio import run_async, silence_asyncio_debug
from aether_paths import REPO_ROOT
from scripts.operations.train_meta_data import (
    META_TRAIN_DEFAULT_BARS,
    OhlcBundle,
    assert_bundles_match_granularity,
    resolve_meta_train_bars,
    resolve_training_bundles,
)
from scripts.operations.train_meta_optuna import (
    META_EXPORT_MIN_ZSCORE,
    assert_export_mae_gap,
    assert_export_zscore_floor,
    configure_meta_train_logging,
    run_optuna_study,
)
from scripts.operations.train_meta_vector import (
    build_paired_training_dataset,
    load_teacher_probs_from_checkpoints,
    resolve_contract_duration_seconds,
)


logger = logging.getLogger("META_TRAIN")
DEFAULT_DSN = "postgresql://aether:aether@localhost:5432/aether"
DEFAULT_OUTPUT = REPO_ROOT / "infra" / "docker" / "meta-models" / "meta_lgbm.pkl"
MIN_TARGET_VARIANCE = 1e-12
DEFAULT_META_TRIALS = 48


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
    return int(data_cfg.get("micro_granularity", 120)) if isinstance(data_cfg, dict) else 120


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
        "oos_payoff_zscore_mean": float(bundle_meta.get("oos_payoff_zscore_mean", 0.0)),
        "oos_information_ratio": float(bundle_meta.get("oos_information_ratio", 0.0)),
        "oos_information_ratio_unit": float(bundle_meta.get("oos_information_ratio_unit", 0.0)),
        "n_val": int(bundle_meta.get("n_val", 0)),
        "optuna_objective_metric": str(bundle_meta.get("optuna_objective_metric", "payoff_zscore")),
    }


def _export_model(model, bundle_meta: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, **bundle_meta}, output_path)


def _teacher_probs(settings: dict[str, Any], symbols: list[str]) -> dict[str, np.ndarray]:
    dl = settings.get("deep_learning") if isinstance(settings.get("deep_learning"), dict) else {}
    template = (
        str(dl.get("model_path_template", "data/dl/{symbol}.pth")) if isinstance(dl, dict) else "data/dl/{symbol}.pth"
    )
    lookback = int(dl.get("lookback", 72)) if isinstance(dl, dict) else 72
    path_template = template if Path(template).is_absolute() else str(REPO_ROOT / template)
    loaded = load_teacher_probs_from_checkpoints(
        symbols,
        model_path_template=path_template,
        lookback=lookback,
        repo_root=REPO_ROOT,
    )
    if loaded:
        logger.info("META_TRAIN: teacher TCN carregado para %s", ",".join(sorted(loaded)))
    else:
        logger.warning("META_TRAIN: teacher TCN ausente; usando proxy de retorno passado.")
    return loaded


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
    export_min_zscore: float = META_EXPORT_MIN_ZSCORE,
) -> dict[str, Any]:
    required_gran = int(granularity)
    bundles = await resolve_training_bundles(
        settings=settings,
        dsn=dsn,
        symbols=symbols,
        granularity=required_gran,
        bars=bars,
        source=source,
        require_exact_granularity=True,
    )
    assert_bundles_match_granularity(bundles, required_gran)
    micro_gran = _micro_granularity(settings)
    contract_duration = resolve_contract_duration_seconds(settings)
    teacher = _teacher_probs(settings, symbols)
    frame, y, _proxy, _pnl = build_paired_training_dataset(
        bundles,
        micro_granularity=micro_gran,
        contract_duration_seconds=contract_duration,
        fetch_count=resolve_meta_train_bars(bars),
        teacher_probs=teacher or None,
    )
    validate_target_variance(y)
    model, bundle_meta, train_mae, val_mae = run_optuna_study(
        frame,
        y,
        trials=trials,
        granularity=required_gran,
    )
    assert_export_zscore_floor(bundle_meta, floor=float(export_min_zscore))
    assert_export_mae_gap(train_mae, val_mae)
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


def _parse_args(settings: dict[str, Any]) -> argparse.Namespace:
    default_gran = _micro_granularity(settings)
    parser = argparse.ArgumentParser(description="Treina meta-regressor tabular com Optuna.")
    parser.add_argument("--trials", type=int, default=DEFAULT_META_TRIALS)
    parser.add_argument("--granularity", type=int, default=default_gran)
    parser.add_argument("--bars", type=int, default=META_TRAIN_DEFAULT_BARS)
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--symbols", nargs="+", default=["RDBULL", "RDBEAR"])
    parser.add_argument("--source", choices=("auto", "timescale", "deriv"), default="auto")
    parser.add_argument("--export-min-zscore", type=float, default=META_EXPORT_MIN_ZSCORE)
    return parser.parse_args()


def main() -> None:
    silence_asyncio_debug()
    logging.basicConfig(level=logging.INFO)
    configure_meta_train_logging()
    settings = _load_settings()
    args = _parse_args(settings)
    dsn = _resolve_dsn(settings)
    summary = run_async(
        train_meta_classifier(
            settings=settings,
            dsn=dsn,
            symbols=[str(s) for s in args.symbols],
            granularity=int(args.granularity),
            bars=int(args.bars),
            trials=int(args.trials),
            output_path=Path(args.output),
            source=str(args.source),
            export_min_zscore=float(args.export_min_zscore),
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
