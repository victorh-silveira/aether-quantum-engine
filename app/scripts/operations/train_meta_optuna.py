"""Montagem de dataset e objetivo Optuna com regressao de payoff continuo."""

from __future__ import annotations

import logging
import math
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import polars as pl
from sklearn.metrics import mean_absolute_error

from scripts.operations.train_meta_vector import build_paired_training_dataset
from src.application.services.deep_learning.dl_features import FEATURE_DIM
from src.application.services.meta_classifier_cross_symbol import (
    CROSS_SYMBOL_FEATURE_COUNT,
    META_FEATURE_DIM,
)
from src.application.services.meta_classifier_features import meta_classifier_column_names


logger = logging.getLogger("AETH.meta")

LGBM_QUIET_PARAMS: dict[str, Any] = {"verbose": -1, "warnings": False, "n_jobs": 2}
OPTUNA_N_JOBS = 2
LGBM_REGRESSION_OBJECTIVE = "huber"
LGBM_N_ESTIMATORS = 1000
OPTUNA_OOS_PAYOFF_ZSCORE_MIN = 0.04
META_EXPORT_MIN_ZSCORE = 0.04
META_EXPORT_MIN_IR = 1.0
OPTUNA_IR_TIEBREAK_WEIGHT = 0.01
META_EXPORT_MAX_MAE_GAP = 2.0
OPTUNA_OVERFIT_PENALTY = -1.0
OPTUNA_NEGATIVE_EDGE_PENALTY = -1.0
PURGED_SPLIT_EMBARGO = 32
LGBM_EARLY_STOPPING_ROUNDS = 50


def configure_meta_train_logging() -> None:
    logging.getLogger("lightgbm").setLevel(logging.ERROR)
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        silent_logger = getattr(lgb, "basic_logger", None)
        if silent_logger is not None and hasattr(lgb, "register_logger"):
            lgb.register_logger(silent_logger.ERROR)
    except (AttributeError, TypeError, ValueError):
        pass


def _feature_frame(frame: pl.DataFrame | np.ndarray) -> pl.DataFrame:
    columns = meta_classifier_column_names()
    if isinstance(frame, pl.DataFrame):
        return frame.select(columns)
    matrix = np.asarray(frame, dtype=np.float64)
    return pl.DataFrame({name: matrix[:, idx] for idx, name in enumerate(columns)}).select(columns)


def train_lgbm_candidate(
    x_train: pl.DataFrame,
    y_train: np.ndarray,
    x_val: pl.DataFrame,
    y_val: np.ndarray,
    params: dict[str, Any],
    *,
    sample_weight: np.ndarray | None = None,
    early_stopping: bool = True,
) -> tuple[lgb.Booster, float, float]:
    columns = meta_classifier_column_names()
    x_train = _feature_frame(x_train)
    x_val = _feature_frame(x_val)
    x_train_np = x_train.to_numpy()
    x_val_np = x_val.to_numpy()
    merged = {**LGBM_QUIET_PARAMS, **params}
    train_set = lgb.Dataset(
        x_train_np,
        label=y_train,
        feature_name=columns,
        weight=np.asarray(sample_weight, dtype=np.float64) if sample_weight is not None else None,
    )
    val_set = lgb.Dataset(x_val_np, label=y_val, reference=train_set)
    model = lgb.train(
        {"objective": LGBM_REGRESSION_OBJECTIVE, "verbosity": -1, "seed": 42, **merged},
        train_set,
        num_boost_round=int(LGBM_N_ESTIMATORS),
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(int(LGBM_EARLY_STOPPING_ROUNDS), verbose=False)] if early_stopping else [],
    )
    train_pred = model.predict(x_train_np)
    val_pred = model.predict(x_val_np)
    train_mae = float(mean_absolute_error(y_train, train_pred))
    val_mae = float(mean_absolute_error(y_val, val_pred))
    return model, train_mae, val_mae


def information_ratio_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula Information Ratio da estratégia simulada com sizing por sinal previsto."""
    scale = np.tanh(np.asarray(y_pred, dtype=np.float64))
    realized = scale * np.asarray(y_true, dtype=np.float64)
    if realized.size == 0:
        return 0.0
    std = float(np.std(realized, ddof=0))
    if std <= 1e-12:
        return 0.0
    mean = float(np.mean(realized))
    return float(mean / std * np.sqrt(realized.size))


def payoff_zscore_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula z-score médio do payoff OOS induzido pelas predições."""
    scale = np.tanh(np.asarray(y_pred, dtype=np.float64))
    realized = scale * np.asarray(y_true, dtype=np.float64)
    std = float(np.std(realized, ddof=0))
    if std <= 1e-12:
        return 0.0
    return float(np.mean(realized) / std)


def _mae_gap_ratio(train_mae: float, val_mae: float) -> float:
    return float(val_mae) / max(float(train_mae), 1e-9)


def _purged_frame_split(
    frame: pl.DataFrame,
    y: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    embargo: int = PURGED_SPLIT_EMBARGO,
) -> tuple[pl.DataFrame, pl.DataFrame, np.ndarray, np.ndarray, np.ndarray | None]:
    sample_count = int(frame.height)
    val_size = max(32, int(sample_count * 0.15))
    train_end = sample_count - val_size - max(0, int(embargo))
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    if train_end < 32 or val_size < 8:
        cut = max(1, int(sample_count * 0.8))
        w_train = None if weights is None else weights[:cut]
        return frame.slice(0, cut), frame.slice(cut, sample_count - cut), y[:cut], y[cut:], w_train
    val_start = train_end + max(0, int(embargo))
    w_train = None if weights is None else weights[:train_end]
    val_frame = frame.slice(val_start, sample_count - val_start)
    val_y = y[val_start:]
    # Subamostragem com passo stride=5 para eliminar sobreposicao temporal de 5 minutos
    step = 5
    val_frame_sub = val_frame.gather_every(step)
    val_y_sub = val_y[::step]
    return (
        frame.slice(0, train_end),
        val_frame_sub,
        y[:train_end],
        val_y_sub,
        w_train,
    )


def _hygiene_for_bundle(hygiene: dict[str, Any]) -> dict[str, Any]:
    """Serializa hygiene misto (int/float/str) para bundle_meta sem forcar int."""
    out: dict[str, Any] = {}
    for key, value in hygiene.items():
        name = str(key)
        if isinstance(value, (bool, int, np.integer)):
            out[name] = int(value)
        elif isinstance(value, (float, np.floating)):
            out[name] = float(value)
        else:
            out[name] = str(value)
    return out


def run_optuna_study(
    frame: pl.DataFrame,
    y: np.ndarray,
    *,
    trials: int,
    granularity: int | None = None,
    sample_weight: np.ndarray | None = None,
    hygiene: dict[str, Any] | None = None,
) -> tuple[lgb.Booster, dict[str, Any], float, float]:
    configure_meta_train_logging()
    frame = _feature_frame(frame)
    x_train, x_val, y_train, y_val, w_train = _purged_frame_split(
        frame,
        y,
        sample_weight=sample_weight,
    )
    n_val = int(x_val.height)
    x_val_np = x_val.to_numpy()

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 4),
            "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 4, 16),
            "min_child_samples": trial.suggest_int("min_child_samples", 30, 150),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 50.0, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.50, 0.75),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
            "n_jobs": OPTUNA_N_JOBS,
        }
        model, train_mae, val_mae = train_lgbm_candidate(
            x_train,
            y_train,
            x_val,
            y_val,
            params,
            sample_weight=w_train,
        )
        gap = _mae_gap_ratio(train_mae, val_mae)
        trial.set_user_attr("mae_gap", float(gap))
        if gap > META_EXPORT_MAX_MAE_GAP + 1e-12:
            return float(OPTUNA_OVERFIT_PENALTY)
        val_pred = model.predict(x_val_np)
        oos_zscore = payoff_zscore_mean(y_val, val_pred)
        oos_ir = information_ratio_from_predictions(y_val, val_pred)
        trial.set_user_attr("oos_payoff_zscore_mean", float(oos_zscore))
        trial.set_user_attr("oos_information_ratio", float(oos_ir))
        if float(oos_zscore) <= 0.0:
            return float(OPTUNA_NEGATIVE_EDGE_PENALTY)
        return float(oos_zscore) + OPTUNA_IR_TIEBREAK_WEIGHT * float(oos_ir)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials, show_progress_bar=False, n_jobs=OPTUNA_N_JOBS)
    if float(study.best_value) <= float(OPTUNA_OVERFIT_PENALTY) + 1e-12:
        raise RuntimeError(
            "Export meta bloqueado: nenhum trial Optuna passou o teto de overfit "
            f"val_mae/train_mae<={META_EXPORT_MAX_MAE_GAP:.1f} ou edge OOS positivo. "
            "Aumente barras/trials ou regularizacao."
        )
    best_trial = study.best_trial
    best_z = float(best_trial.user_attrs.get("oos_payoff_zscore_mean", 0.0))
    best_ir = float(best_trial.user_attrs.get("oos_information_ratio", 0.0) or 0.0)
    assert_export_zscore_floor(
        {"oos_payoff_zscore_mean": best_z, "oos_information_ratio": best_ir},
        floor=float(OPTUNA_OOS_PAYOFF_ZSCORE_MIN),
        min_ir=float(META_EXPORT_MIN_IR),
    )
    best_params = {**study.best_params, "n_jobs": OPTUNA_N_JOBS}
    model, train_mae, val_mae = train_lgbm_candidate(
        x_train,
        y_train,
        x_val,
        y_val,
        best_params,
        sample_weight=w_train,
    )
    if _mae_gap_ratio(train_mae, val_mae) > META_EXPORT_MAX_MAE_GAP + 1e-12:
        raise RuntimeError(
            "Export meta bloqueado: melhor trial ainda overfitou no refit final "
            f"(val_mae/train_mae={_mae_gap_ratio(train_mae, val_mae):.3f})."
        )
    val_pred = model.predict(x_val_np)
    val_ir = information_ratio_from_predictions(y_val, val_pred)
    val_oos_zscore = payoff_zscore_mean(y_val, val_pred)
    ir_unit = float(val_ir / math.sqrt(n_val)) if n_val > 0 else 0.0
    logger.info(
        "Optuna concluido | z=%.3f ir=%.2f n_val=%d mae=%.3f trials=%d depth=%s lr=%s",
        val_oos_zscore,
        val_ir,
        n_val,
        val_mae,
        trials,
        best_params.get("max_depth"),
        f"{float(best_params.get('learning_rate') or 0.0):.4f}",
    )
    bundle_meta = {
        "model_type": "regressor",
        "objective": LGBM_REGRESSION_OBJECTIVE,
        "feature_dim": META_FEATURE_DIM,
        "base_feature_dim": FEATURE_DIM,
        "cross_symbol_feature_count": CROSS_SYMBOL_FEATURE_COUNT,
        "flow_feature_count": 2,
        "feature_names": meta_classifier_column_names(),
        "optuna_objective_metric": "payoff_zscore",
        "oos_payoff_zscore_mean": float(val_oos_zscore),
        "oos_information_ratio": float(val_ir),
        "oos_information_ratio_unit": float(ir_unit),
        "n_val": int(n_val),
        "granularity": int(granularity) if granularity is not None else None,
        **best_params,
    }
    if hygiene:
        bundle_meta.update(_hygiene_for_bundle(hygiene))
    return model, bundle_meta, train_mae, val_mae


def assert_export_zscore_floor(
    bundle_meta: dict[str, Any],
    *,
    floor: float = META_EXPORT_MIN_ZSCORE,
    min_ir: float = META_EXPORT_MIN_IR,
) -> None:
    zscore = float(bundle_meta.get("oos_payoff_zscore_mean", 0.0))
    ir = float(bundle_meta.get("oos_information_ratio", 0.0) or 0.0)
    if zscore + 1e-12 >= float(floor):
        return
    if ir + 1e-12 >= float(min_ir):
        return
    raise RuntimeError(
        f"Export meta bloqueado: oos_payoff_zscore_mean={zscore:.6f} < floor={float(floor):.6f} "
        f"e oos_information_ratio={ir:.6f} < min_ir={float(min_ir):.6f}. "
        "Retreine com teacher TCN (data/dl), gran=60s, mais barras/trials ou features alinhadas ao runtime."
    )


def assert_export_mae_gap(
    train_mae: float,
    val_mae: float,
    *,
    max_gap: float = META_EXPORT_MAX_MAE_GAP,
) -> None:
    train = max(float(train_mae), 1e-9)
    ratio = float(val_mae) / train
    if ratio <= float(max_gap) + 1e-12:
        return
    raise RuntimeError(
        f"Export meta bloqueado: val_mae/train_mae={ratio:.3f} > max_gap={float(max_gap):.3f}. "
        "Overfit tabular; aumente embargo/purge, barras ou regularizacao."
    )


__all__ = [
    "LGBM_QUIET_PARAMS",
    "LGBM_REGRESSION_OBJECTIVE",
    "META_EXPORT_MAX_MAE_GAP",
    "META_EXPORT_MIN_IR",
    "META_EXPORT_MIN_ZSCORE",
    "OPTUNA_NEGATIVE_EDGE_PENALTY",
    "OPTUNA_OOS_PAYOFF_ZSCORE_MIN",
    "OPTUNA_N_JOBS",
    "assert_export_mae_gap",
    "assert_export_zscore_floor",
    "information_ratio_from_predictions",
    "payoff_zscore_mean",
    "build_paired_training_dataset",
    "configure_meta_train_logging",
    "run_optuna_study",
    "train_lgbm_candidate",
]
