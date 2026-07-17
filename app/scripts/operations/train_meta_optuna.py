"""Montagem de dataset e objetivo Optuna com regressao de payoff continuo."""

from __future__ import annotations

import logging
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error

from scripts.operations.train_meta_vector import build_paired_training_dataset
from src.application.services.deep_learning.dl_features import FEATURE_DIM
from src.application.services.meta_classifier_cross_symbol import (
    CROSS_SYMBOL_FEATURE_COUNT,
    META_FEATURE_DIM,
)
from src.application.services.meta_classifier_features import meta_classifier_column_names


logger = logging.getLogger("META_TRAIN")

LGBM_QUIET_PARAMS: dict[str, Any] = {"verbose": -1, "warnings": False, "n_jobs": 2}
OPTUNA_N_JOBS = 2
LGBM_REGRESSION_OBJECTIVE = "huber"
OPTUNA_OOS_PAYOFF_ZSCORE_MIN = 1.0


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


def train_lgbm_candidate(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    params: dict[str, Any],
) -> tuple[lgb.LGBMRegressor, float, float]:
    columns = meta_classifier_column_names()
    x_train = pd.DataFrame(x_train, columns=columns).loc[:, columns]
    x_val = pd.DataFrame(x_val, columns=columns).loc[:, columns]
    merged = {**LGBM_QUIET_PARAMS, **params}
    model = lgb.LGBMRegressor(
        objective=LGBM_REGRESSION_OBJECTIVE,
        n_estimators=180,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        **merged,
    )
    model.fit(x_train, y_train, feature_name=columns)
    train_pred = model.predict(x_train.loc[:, columns])
    val_pred = model.predict(x_val.loc[:, columns])
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


def run_optuna_study(
    frame: pd.DataFrame,
    y: np.ndarray,
    *,
    trials: int,
) -> tuple[lgb.LGBMRegressor, dict[str, Any], float, float]:
    configure_meta_train_logging()
    columns = meta_classifier_column_names()
    frame = pd.DataFrame(frame, columns=columns).loc[:, columns]
    split = int(len(frame) * 0.8)
    x_train, x_val = frame.iloc[:split], frame.iloc[split:]
    y_train, y_val = y[:split], y[split:]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 96),
            "n_jobs": OPTUNA_N_JOBS,
        }
        model, _, _ = train_lgbm_candidate(x_train, y_train, x_val, y_val, params)
        val_pred = model.predict(x_val.loc[:, columns])
        oos_zscore = payoff_zscore_mean(y_val, val_pred)
        trial.set_user_attr("oos_payoff_zscore_mean", float(oos_zscore))
        if oos_zscore < OPTUNA_OOS_PAYOFF_ZSCORE_MIN:
            return -1_000_000.0 + float(oos_zscore)
        return information_ratio_from_predictions(y_val, val_pred)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials, show_progress_bar=False, n_jobs=OPTUNA_N_JOBS)
    best_params = {**study.best_params, "n_jobs": OPTUNA_N_JOBS}
    model, train_mae, val_mae = train_lgbm_candidate(x_train, y_train, x_val, y_val, best_params)
    val_pred = model.predict(x_val.loc[:, columns])
    val_ir = information_ratio_from_predictions(y_val, val_pred)
    val_oos_zscore = payoff_zscore_mean(y_val, val_pred)
    logger.info(
        "Optuna concluido | val_ir=%.6f | val_zscore=%.6f | val_mae=%.6f | train_mae=%.6f | params=%s | trials=%d",
        val_ir,
        val_oos_zscore,
        val_mae,
        train_mae,
        best_params,
        trials,
    )
    bundle_meta = {
        "model_type": "regressor",
        "objective": LGBM_REGRESSION_OBJECTIVE,
        "feature_dim": META_FEATURE_DIM,
        "base_feature_dim": FEATURE_DIM,
        "cross_symbol_feature_count": CROSS_SYMBOL_FEATURE_COUNT,
        "flow_feature_count": 2,
        "feature_names": meta_classifier_column_names(),
        "optuna_objective_metric": "information_ratio",
        "oos_payoff_zscore_mean": float(val_oos_zscore),
        **best_params,
    }
    return model, bundle_meta, train_mae, val_mae


__all__ = [
    "LGBM_QUIET_PARAMS",
    "LGBM_REGRESSION_OBJECTIVE",
    "OPTUNA_OOS_PAYOFF_ZSCORE_MIN",
    "OPTUNA_N_JOBS",
    "information_ratio_from_predictions",
    "payoff_zscore_mean",
    "build_paired_training_dataset",
    "configure_meta_train_logging",
    "run_optuna_study",
    "train_lgbm_candidate",
]
