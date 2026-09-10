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
from sklearn.model_selection import TimeSeriesSplit

from scripts.operations.train_meta_vector import (
    TARGET_NULL_MAE_GAP_MAX,
    _assert_train_val_target_scale,
    _null_mae_gap,
    _scale_targets_from_train,
    _target_std,
    build_paired_training_dataset,
)
from src.application.services.deep_learning.dl_features import FEATURE_DIM
from src.application.services.meta_classifier_cross_symbol import (
    CROSS_SYMBOL_FEATURE_COUNT,
    META_FEATURE_DIM,
)
from src.application.services.meta_classifier_features import meta_classifier_column_names


logger = logging.getLogger("AETH.meta")

LGBM_QUIET_PARAMS: dict[str, Any] = {"verbose": -1, "warnings": False, "n_jobs": 1}
OPTUNA_N_JOBS = 1
LGBM_REGRESSION_OBJECTIVE = "regression_l1"
LGBM_METRIC = "l1"
LGBM_N_ESTIMATORS_LARGE = 80
LGBM_N_ESTIMATORS_CV = 200
OPTUNA_OOS_PAYOFF_ZSCORE_MIN = 0.04
META_EXPORT_MIN_ZSCORE = 0.04
META_EXPORT_MIN_IR = 1.0
OPTUNA_IR_TIEBREAK_WEIGHT = 0.01
META_EXPORT_MAX_MAE_GAP = TARGET_NULL_MAE_GAP_MAX
OPTUNA_OVERFIT_PENALTY = -1.0
OPTUNA_NEGATIVE_EDGE_PENALTY = -1.0
PURGED_SPLIT_EMBARGO = 32
LGBM_EARLY_STOPPING_ROUNDS = 15


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


def _boost_round_budget(*, use_cv: bool) -> int:
    return int(LGBM_N_ESTIMATORS_CV if use_cv else LGBM_N_ESTIMATORS_LARGE)


def _booster_best_iteration(model: Any) -> int:
    selected = getattr(model, "_aether_export_iteration", None)
    if selected is not None:
        try:
            return max(0, int(selected))
        except (TypeError, ValueError):
            pass
    raw = getattr(model, "best_iteration", 0)
    try:
        value = int(raw or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, value)


def _positive_iteration_count(raw: Any) -> int:
    if isinstance(raw, (bool, np.bool_)):
        return 0
    if isinstance(raw, (int, np.integer)):
        value = int(raw)
        return value if value > 0 else 0
    if isinstance(raw, (float, np.floating)) and np.isfinite(raw):
        value = int(raw)
        return value if value > 0 else 0
    return 0


def _booster_trained_iterations(model: Any) -> int:
    current = getattr(model, "current_iteration", None)
    if callable(current):
        try:
            counted = _positive_iteration_count(current())
        except (TypeError, ValueError):
            counted = 0
        if counted > 0:
            return counted
    trees = getattr(model, "num_trees", None)
    if callable(trees):
        try:
            counted = _positive_iteration_count(trees())
        except (TypeError, ValueError):
            counted = 0
        if counted > 0:
            return counted
    return 0


def _finite_number(raw: Any) -> float | None:
    if isinstance(raw, (bool, np.bool_)):
        return None
    if isinstance(raw, (int, float, np.integer, np.floating)) and np.isfinite(raw):
        return float(raw)
    return None


def _label_location(y_train: np.ndarray) -> float:
    arr = np.asarray(y_train, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.median(arr))


def _constant_mae(y: np.ndarray, loc: float) -> float:
    return float(mean_absolute_error(y, np.full(len(y), loc, dtype=np.float64)))


def _predict_with_export(model: Any, x: np.ndarray) -> np.ndarray:
    n = int(np.asarray(x).shape[0])
    chosen = _finite_number(getattr(model, "_aether_export_iteration", None))
    loc = _finite_number(getattr(model, "_aether_label_location", None))
    if loc is None:
        loc = 0.0
    if chosen is not None and int(chosen) <= 0:
        return np.full(n, loc, dtype=np.float64)
    kwargs = {} if chosen is None else {"num_iteration": int(chosen)}
    return np.asarray(model.predict(x, **kwargs), dtype=np.float64)


def _mae_curves_from_booster(
    model: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    n_iter: int,
    loc: float = 0.0,
) -> tuple[list[float], list[float]]:
    train_hist: list[float] = [_constant_mae(y_train, loc)]
    val_hist: list[float] = [_constant_mae(y_val, loc)]
    for idx in range(1, max(1, int(n_iter)) + 1):
        train_hist.append(float(mean_absolute_error(y_train, model.predict(x_train, num_iteration=idx))))
        val_hist.append(float(mean_absolute_error(y_val, model.predict(x_val, num_iteration=idx))))
    return train_hist, val_hist


def _eval_l1_histories(evals_result: dict[str, Any]) -> tuple[list[float], list[float]]:
    def _metric(block: Any) -> list[float]:
        if not isinstance(block, dict) or not block:
            return []
        for key in ("l1", "mae", "regression_l1"):
            raw = block.get(key)
            if raw is not None:
                return [float(v) for v in raw]
        first = next(iter(block.values()))
        return [float(v) for v in first]

    payload = evals_result if isinstance(evals_result, dict) else {}
    return _metric(payload.get("train")), _metric(payload.get("valid"))


def _select_gap_feasible_iteration(
    train_hist: list[float],
    val_hist: list[float],
    *,
    max_gap: float = META_EXPORT_MAX_MAE_GAP,
) -> int | None:
    if not train_hist or not val_hist:
        return None
    count = min(len(train_hist), len(val_hist))
    feasible = [
        idx
        for idx in range(count)
        if float(val_hist[idx]) / max(float(train_hist[idx]), 1e-9) <= float(max_gap) + 1e-12
    ]
    if not feasible:
        return 0
    best = min(feasible, key=lambda idx: (float(val_hist[idx]), idx))
    return int(best)


def train_lgbm_candidate(
    x_train: pl.DataFrame,
    y_train: np.ndarray,
    x_val: pl.DataFrame,
    y_val: np.ndarray,
    params: dict[str, Any],
    *,
    sample_weight: np.ndarray | None = None,
    early_stopping: bool = True,
    num_boost_round: int | None = None,
) -> tuple[lgb.Booster, float, float]:
    columns = meta_classifier_column_names()
    x_train = _feature_frame(x_train)
    x_val = _feature_frame(x_val)
    x_train_np = np.ascontiguousarray(x_train.to_numpy(), dtype=np.float64)
    x_val_np = np.ascontiguousarray(x_val.to_numpy(), dtype=np.float64)
    y_train_arr = np.ascontiguousarray(y_train, dtype=np.float64)
    y_val_arr = np.ascontiguousarray(y_val, dtype=np.float64)
    loc = _label_location(y_train_arr)
    rounds = int(num_boost_round) if num_boost_round is not None else int(LGBM_N_ESTIMATORS_LARGE)
    merged = {**LGBM_QUIET_PARAMS, **params, "metric": LGBM_METRIC}
    train_set = lgb.Dataset(
        x_train_np,
        label=y_train_arr,
        feature_name=columns,
        weight=np.ascontiguousarray(sample_weight, dtype=np.float64) if sample_weight is not None else None,
        free_raw_data=False,
    )
    val_set = lgb.Dataset(x_val_np, label=y_val_arr, reference=train_set, free_raw_data=False)
    evals_result: dict[str, Any] = {}
    callbacks: list[Any] = [lgb.record_evaluation(evals_result)]
    if early_stopping:
        callbacks.append(lgb.early_stopping(int(LGBM_EARLY_STOPPING_ROUNDS), verbose=False))
    model = lgb.train(
        {"objective": LGBM_REGRESSION_OBJECTIVE, "verbosity": -1, "seed": 42, **merged},
        train_set,
        num_boost_round=rounds,
        valid_sets=[train_set, val_set],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )
    train_hist, val_hist = _eval_l1_histories(evals_result)
    n_iter = _booster_trained_iterations(model)
    if type(model).__name__ == "Booster":
        n_iter = max(n_iter, int(rounds))
        train_hist, val_hist = _mae_curves_from_booster(
            model,
            x_train_np,
            y_train_arr,
            x_val_np,
            y_val_arr,
            n_iter=n_iter,
            loc=loc,
        )
    export_iter = _select_gap_feasible_iteration(train_hist, val_hist)
    try:
        model._aether_label_location = float(loc)
        if export_iter is not None:
            model._aether_export_iteration = int(export_iter)
    except (AttributeError, TypeError, ValueError):
        pass
    train_pred = _predict_with_export(model, x_train_np)
    val_pred = _predict_with_export(model, x_val_np)
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


def _lgbm_search_bounds(train_rows: int, *, use_cv: bool) -> tuple[int, int, int, float]:
    min_child_hi = max(4, min(180, int(train_rows) // 4))
    if use_cv:
        min_child_lo = max(2, min(40, min_child_hi // 2))
        return min(min_child_lo, min_child_hi), min_child_hi, 4, 0.1
    min_child_lo = max(32, min(80, min_child_hi // 2))
    return min(min_child_lo, min_child_hi), min_child_hi, 1, 16.0


def _set_mae_user_attrs(
    trial: optuna.Trial,
    train_mae: float,
    val_mae: float,
    gap: float,
    *,
    best_iteration: int = 0,
) -> None:
    trial.set_user_attr("mae_gap", float(gap))
    trial.set_user_attr("train_mae", float(train_mae))
    trial.set_user_attr("val_mae", float(val_mae))
    trial.set_user_attr("best_iteration", int(best_iteration))


def _median_trial_attr(trials: list[Any], key: str) -> float:
    vals = [float(trial.user_attrs[key]) for trial in trials if trial.user_attrs.get(key) is not None]
    if not vals:
        return float("nan")
    return float(np.median(np.asarray(vals, dtype=np.float64)))


def _count_blocked_trials(study: optuna.Study) -> tuple[int, int]:
    n_overfit = 0
    n_neg_edge = 0
    for trial in study.trials:
        gap = trial.user_attrs.get("mae_gap")
        z = trial.user_attrs.get("oos_payoff_zscore_mean")
        if gap is not None and float(gap) > META_EXPORT_MAX_MAE_GAP + 1e-12:
            n_overfit += 1
        elif z is not None and float(z) <= 0.0:
            n_neg_edge += 1
        elif float(trial.value or OPTUNA_OVERFIT_PENALTY) <= float(OPTUNA_OVERFIT_PENALTY) + 1e-12:
            n_overfit += 1
    return n_overfit, n_neg_edge


def _format_optuna_blocked_error(
    *,
    sample_count: int,
    n_overfit: int,
    n_neg_edge: int,
    trials: int,
    y: np.ndarray,
    hygiene: dict[str, Any] | None,
    study: optuna.Study,
) -> str:
    label_mode = int((hygiene or {}).get("label_mode", -1))
    arr = np.asarray(y, dtype=np.float64)
    y_std = float(np.std(arr, ddof=0)) if arr.size else 0.0
    med_gap = _median_trial_attr(study.trials, "mae_gap")
    med_tr = _median_trial_attr(study.trials, "train_mae")
    med_va = _median_trial_attr(study.trials, "val_mae")
    med_iter = _median_trial_attr(study.trials, "best_iteration")
    naive_mae = float(np.mean(np.abs(arr))) if arr.size else 0.0
    train_std = float((hygiene or {}).get("train_std", float("nan")))
    val_std = float((hygiene or {}).get("val_std", float("nan")))
    return (
        "Export meta bloqueado: nenhum trial Optuna passou o teto de overfit "
        f"val_mae/train_mae<={META_EXPORT_MAX_MAE_GAP:.1f} ou edge OOS positivo "
        f"(n={sample_count} overfit={n_overfit} neg_edge={n_neg_edge} trials={trials} "
        f"label_mode={label_mode} y_std={y_std:.4f} "
        f"train_std={train_std:.4f} val_std={val_std:.4f} "
        f"med_mae_gap={med_gap:.3f} med_train_mae={med_tr:.4f} med_val_mae={med_va:.4f} "
        f"med_best_iter={med_iter:.0f} med_naive_mae={naive_mae:.4f}). "
        "Alvo payoff unit-scaled O(1) no fold de treino; objetivo L1; nao baixe o teto 2.0."
    )


def _purged_frame_split(
    frame: pl.DataFrame,
    y: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    embargo: int = PURGED_SPLIT_EMBARGO,
) -> tuple[pl.DataFrame, pl.DataFrame, np.ndarray, np.ndarray, np.ndarray | None]:
    sample_count = int(frame.height)
    val_size = max(32, int(sample_count * 0.25))
    train_end = sample_count - val_size - max(0, int(embargo))
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    if train_end < 32 or val_size < 8:
        cut = max(1, int(sample_count * 0.8))
        w_train = None if weights is None else weights[:cut]
        return frame.slice(0, cut), frame.slice(cut, sample_count - cut), y[:cut], y[cut:], w_train
    val_start = train_end + max(0, int(embargo))
    w_train = None if weights is None else weights[:train_end]
    val_frame = frame.slice(val_start, sample_count - val_start)
    return (
        frame.slice(0, train_end),
        val_frame,
        y[:train_end],
        y[val_start:],
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
    sample_count = int(frame.height)
    x_train, x_val, y_train, y_val, w_train = _purged_frame_split(
        frame,
        y,
        sample_weight=sample_weight,
    )
    train_std = _target_std(y_train)
    val_std = _target_std(y_val)
    _assert_train_val_target_scale(
        train_std,
        val_std,
        null_mae_gap=_null_mae_gap(y_train, y_val),
    )
    y_train, y_val, label_scale = _scale_targets_from_train(y_train, y_val)
    if hygiene is not None:
        hygiene["label_scale"] = float(label_scale)
        hygiene["train_std"] = float(train_std)
        hygiene["val_std"] = float(val_std)
    y_fit = np.concatenate([np.asarray(y_train, dtype=np.float64), np.asarray(y_val, dtype=np.float64)])
    n_val = int(x_val.height)
    x_val_np = x_val.to_numpy()

    train_rows = int(x_train.height)
    use_cv = sample_count < 150
    min_child_lo, min_child_hi, depth_hi, lambda_lo = _lgbm_search_bounds(train_rows, use_cv=use_cv)
    boost_rounds = _boost_round_budget(use_cv=use_cv)
    lr_hi = 0.12 if use_cv else 0.02
    leaves_hi = 10 if use_cv else 2
    if use_cv:
        cv_splits = 3
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        frame_np = frame.to_numpy()

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 1, depth_hi),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, lr_hi, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, leaves_hi),
            "min_child_samples": trial.suggest_int("min_child_samples", min_child_lo, min_child_hi),
            "reg_lambda": trial.suggest_float("reg_lambda", lambda_lo, 100.0, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.40, 0.85),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.50, 0.80),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
            "n_jobs": OPTUNA_N_JOBS,
        }
        if use_cv:
            fold_z, fold_ir, fold_gaps, fold_tr, fold_va, fold_iter = [], [], [], [], [], []
            for tr_idx, v_idx in tscv.split(frame_np):
                f_tr, f_v = frame[tr_idx], frame[v_idx]
                y_tr, y_v, _ = _scale_targets_from_train(y[tr_idx], y[v_idx])
                w_tr_fold = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)[tr_idx]
                m_fold, tr_mae, val_mae = train_lgbm_candidate(
                    f_tr,
                    y_tr,
                    f_v,
                    y_v,
                    params,
                    sample_weight=w_tr_fold,
                    num_boost_round=boost_rounds,
                )
                gap = _mae_gap_ratio(tr_mae, val_mae)
                best_iter = _booster_best_iteration(m_fold)
                _set_mae_user_attrs(trial, tr_mae, val_mae, gap, best_iteration=best_iter)
                if gap > META_EXPORT_MAX_MAE_GAP + 1e-12:
                    return float(OPTUNA_OVERFIT_PENALTY)
                fold_gaps.append(gap)
                fold_tr.append(float(tr_mae))
                fold_va.append(float(val_mae))
                fold_iter.append(float(best_iter))
                preds = _predict_with_export(m_fold, f_v.to_numpy())
                fold_z.append(payoff_zscore_mean(y_v, preds))
                fold_ir.append(information_ratio_from_predictions(y_v, preds))
            mean_z = float(np.mean(fold_z))
            mean_ir = float(np.mean(fold_ir))
            _set_mae_user_attrs(
                trial,
                float(np.mean(fold_tr)),
                float(np.mean(fold_va)),
                float(np.mean(fold_gaps)),
                best_iteration=int(round(float(np.mean(fold_iter)))),
            )
            trial.set_user_attr("oos_payoff_zscore_mean", mean_z)
            trial.set_user_attr("oos_information_ratio", mean_ir)
            if mean_z <= 0.0:
                return float(OPTUNA_NEGATIVE_EDGE_PENALTY)
            return mean_z + OPTUNA_IR_TIEBREAK_WEIGHT * mean_ir

        model, train_mae, val_mae = train_lgbm_candidate(
            x_train,
            y_train,
            x_val,
            y_val,
            params,
            sample_weight=w_train,
            num_boost_round=boost_rounds,
        )
        gap = _mae_gap_ratio(train_mae, val_mae)
        _set_mae_user_attrs(
            trial,
            train_mae,
            val_mae,
            gap,
            best_iteration=_booster_best_iteration(model),
        )
        if gap > META_EXPORT_MAX_MAE_GAP + 1e-12:
            return float(OPTUNA_OVERFIT_PENALTY)
        val_pred = _predict_with_export(model, x_val_np)
        oos_zscore = payoff_zscore_mean(y_val, val_pred)
        oos_ir = information_ratio_from_predictions(y_val, val_pred)
        trial.set_user_attr("oos_payoff_zscore_mean", float(oos_zscore))
        trial.set_user_attr("oos_information_ratio", float(oos_ir))
        if float(oos_zscore) <= 0.0:
            return float(OPTUNA_NEGATIVE_EDGE_PENALTY)
        confidence_scale = min(1.0, math.sqrt(max(1, n_val) / 64.0))
        return (float(oos_zscore) + OPTUNA_IR_TIEBREAK_WEIGHT * float(oos_ir)) * confidence_scale

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials, show_progress_bar=False, n_jobs=OPTUNA_N_JOBS)
    if float(study.best_value) <= float(OPTUNA_OVERFIT_PENALTY) + 1e-12:
        n_overfit, n_neg_edge = _count_blocked_trials(study)
        raise RuntimeError(
            _format_optuna_blocked_error(
                sample_count=sample_count,
                n_overfit=n_overfit,
                n_neg_edge=n_neg_edge,
                trials=trials,
                y=y_fit,
                hygiene=hygiene
                if hygiene is not None
                else {
                    "train_std": float(train_std),
                    "val_std": float(val_std),
                    "label_scale": float(label_scale),
                },
                study=study,
            )
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
        num_boost_round=boost_rounds,
    )
    if not use_cv and _mae_gap_ratio(train_mae, val_mae) > META_EXPORT_MAX_MAE_GAP + 1e-12:
        raise RuntimeError(
            "Export meta bloqueado: melhor trial ainda overfitou no refit final "
            f"(val_mae/train_mae={_mae_gap_ratio(train_mae, val_mae):.3f})."
        )
    val_pred = _predict_with_export(model, x_val_np)
    val_ir = best_ir if use_cv else information_ratio_from_predictions(y_val, val_pred)
    val_oos_zscore = best_z if use_cv else payoff_zscore_mean(y_val, val_pred)
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
        "label_scale": float(label_scale),
        "train_std": float(train_std),
        "val_std": float(val_std),
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
    "LGBM_METRIC",
    "LGBM_N_ESTIMATORS_CV",
    "LGBM_N_ESTIMATORS_LARGE",
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
