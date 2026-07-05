"""Montagem de dataset e objetivo Optuna com P&L ponderado cross-symbol."""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score

from scripts.operations.train_meta_data import OhlcBundle
from src.application.services.deep_learning.dl_feature_build import build_feature_matrix, precompute_price_series
from src.application.services.deep_learning.dl_features import FEATURE_DIM
from src.application.services.meta_classifier_cross_symbol import (
    CROSS_SYMBOL_FEATURE_COUNT,
    META_FEATURE_DIM,
    compute_cross_symbol_triplet,
)
from src.application.services.meta_classifier_features import meta_classifier_column_names
from src.application.services.meta_classifier_flow_features import flow_features_from_micro_series


def _micro_reversal_label(tcn_prob: float, forward_delta: float) -> int:
    """Rotula payoff micro de 60s sob otica de reversao quando TCN perde e inversao vence."""
    tcn_call = float(tcn_prob) >= 0.5
    tcn_won = (forward_delta > 0.0) if tcn_call else (forward_delta < 0.0)
    invert_won = (forward_delta < 0.0) if tcn_call else (forward_delta > 0.0)
    if not tcn_won and invert_won:
        return 1
    if tcn_won:
        return 1
    return 0


def _proxy_prob_from_forward(forward: float) -> float:
    return float(np.clip(0.5 + 0.15 * forward, 0.05, 0.95))


def _micro_tail_indicators(closes: np.ndarray, *, symbol: str, granularity: int) -> tuple[float, float]:
    series = precompute_price_series(closes, granularity=granularity, symbol=symbol)
    rsi = float(series["rsi"][-1]) if len(series.get("rsi", [])) > 0 else 0.0
    vol_ratio = float(series["vol_ratio_short_long"][-1]) if len(series.get("vol_ratio_short_long", [])) > 0 else 0.0
    return rsi, vol_ratio


def _bundle_frame(bundle: OhlcBundle) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    series = precompute_price_series(
        bundle.closes,
        granularity=bundle.granularity,
        symbol=bundle.symbol,
        open_=bundle.open_,
        high=bundle.high,
        low=bundle.low,
    )
    features = build_feature_matrix(series)
    forward = np.zeros(len(bundle.closes), dtype=np.float32)
    forward[:-1] = bundle.closes[1:] - bundle.closes[:-1]
    pnl = forward.copy()
    labels = (forward > 0.0).astype(np.int32)
    proxy = np.array([_proxy_prob_from_forward(float(v)) for v in forward], dtype=np.float32)
    return features, labels, proxy, pnl


def build_paired_training_dataset(
    bundles: list[OhlcBundle],
    *,
    micro_granularity: int = 60,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Alinha RDBULL/RDBEAR e anexa features cross-symbol com pesos de P&L."""
    by_symbol = {bundle.symbol: bundle for bundle in bundles}
    bull = by_symbol.get("RDBULL")
    bear = by_symbol.get("RDBEAR")
    if bull is None or bear is None:
        raise RuntimeError("Treino meta-classificador exige bundles RDBULL e RDBEAR alinhados.")
    bull_x, bull_y, bull_proxy, bull_pnl = _bundle_frame(bull)
    bear_x, bear_y, bear_proxy, bear_pnl = _bundle_frame(bear)
    rows = min(len(bull_x), len(bear_x)) - 34
    if rows <= 0:
        raise RuntimeError("Historico insuficiente para parear features cross-symbol.")
    start = 32
    end = start + rows
    bull_rsi_series, bull_vol_series = [], []
    bear_rsi_series, bear_vol_series = [], []
    for idx in range(start, end):
        bull_slice = bull.closes[: idx + 1]
        bear_slice = bear.closes[: idx + 1]
        bull_rsi, bull_vol = _micro_tail_indicators(bull_slice, symbol="RDBULL", granularity=micro_granularity)
        bear_rsi, bear_vol = _micro_tail_indicators(bear_slice, symbol="RDBEAR", granularity=micro_granularity)
        bull_rsi_series.append(bull_rsi)
        bull_vol_series.append(bull_vol)
        bear_rsi_series.append(bear_rsi)
        bear_vol_series.append(bear_vol)
    feature_rows: list[list[float]] = []
    label_rows: list[int] = []
    proxy_rows: list[float] = []
    pnl_rows: list[float] = []
    for offset, idx in enumerate(range(start, end)):
        bull_metrics = {
            "calibrated_prob": float(bull_proxy[idx]),
            "indicators": {"rsi": bull_rsi_series[offset], "vol_ratio": bull_vol_series[offset]},
            "micro_indicators": {"rsi": bull_rsi_series[offset], "vol_ratio": bull_vol_series[offset]},
        }
        bear_metrics = {
            "calibrated_prob": float(bear_proxy[idx]),
            "indicators": {"rsi": bear_rsi_series[offset], "vol_ratio": bear_vol_series[offset]},
            "micro_indicators": {"rsi": bear_rsi_series[offset], "vol_ratio": bear_vol_series[offset]},
        }
        cross = compute_cross_symbol_triplet(bull_metrics, bear_metrics)
        bull_slice = bull.closes[: idx + 1]
        flow = flow_features_from_micro_series(
            bull_slice,
            granularity=micro_granularity,
            symbol="RDBULL",
            open_=bull.open_[: idx + 1] if bull.open_ is not None else None,
            high=bull.high[: idx + 1] if bull.high is not None else None,
            low=bull.low[: idx + 1] if bull.low is not None else None,
        )
        row = np.concatenate(
            [
                bull_x[idx],
                np.asarray(list(cross.values()), dtype=np.float32),
                np.asarray(list(flow.values()), dtype=np.float32),
            ]
        ).tolist()
        if len(row) != META_FEATURE_DIM:
            raise RuntimeError(f"Meta feature row divergente: esperado {META_FEATURE_DIM}, obtido {len(row)}")
        feature_rows.append(row)
        label_rows.append(_micro_reversal_label(float(bull_proxy[idx]), float(bull_pnl[idx])))
        proxy_rows.append(float(bull_proxy[idx]))
        pnl_rows.append(float(bull_pnl[idx]))
    frame = pd.DataFrame(feature_rows, columns=meta_classifier_column_names())
    return (
        frame,
        np.asarray(label_rows, dtype=np.int32),
        np.asarray(proxy_rows, dtype=np.float32),
        np.asarray(pnl_rows, dtype=np.float32),
    )


def weighted_pnl_edge_score(
    y_true: np.ndarray,
    prob: np.ndarray,
    proxy_prob: np.ndarray,
    pnl: np.ndarray,
    cross_prob_delta: np.ndarray,
) -> float:
    """Objetivo Optuna ponderando payoff historico, cross-symbol e reversao micro."""
    blended = 0.45 * prob + 0.55 * proxy_prob
    tcn_call = (proxy_prob >= 0.5).astype(np.int32)
    invert_call = 1 - tcn_call
    direction = np.where(blended < 0.42, invert_call, tcn_call)
    signed = np.where(direction == y_true, 1.0, -1.0)
    tcn_won = np.where(
        ((pnl > 0.0) & (tcn_call == 1)) | ((pnl < 0.0) & (tcn_call == 0)),
        1.0,
        0.0,
    )
    invert_won = np.where(
        ((pnl < 0.0) & (tcn_call == 1)) | ((pnl > 0.0) & (tcn_call == 0)),
        1.0,
        0.0,
    )
    reversal_boost = np.where((tcn_won < 0.5) & (invert_won > 0.5), 1.75, 1.0)
    pnl_weight = np.abs(pnl) + 1e-6
    cross_weight = 1.0 + 0.35 * np.clip(cross_prob_delta, 0.0, 1.0)
    payoff = signed * pnl_weight * cross_weight * reversal_boost
    return float(np.mean(payoff))


def train_lgbm_candidate(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    proxy_val: np.ndarray,
    pnl_val: np.ndarray,
    cross_val: np.ndarray,
    params: dict[str, Any],
) -> tuple[lgb.LGBMClassifier, float]:
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=180,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        **params,
    )
    model.fit(x_train, y_train)
    prob = model.predict_proba(x_val)[:, 1]
    edge = weighted_pnl_edge_score(y_val, prob, proxy_val, pnl_val, cross_val)
    auc = roc_auc_score(y_val, prob) if len(np.unique(y_val)) > 1 else 0.5
    return model, edge + 0.05 * auc


def run_optuna_study(
    frame: pd.DataFrame,
    y: np.ndarray,
    proxy: np.ndarray,
    pnl: np.ndarray,
    *,
    trials: int,
) -> tuple[lgb.LGBMClassifier, dict[str, Any], float]:
    cross_col = frame["cross_symbol_prob_delta"].to_numpy(dtype=np.float32)
    split = int(len(frame) * 0.8)
    x_train, x_val = frame.iloc[:split], frame.iloc[split:]
    y_train, y_val = y[:split], y[split:]
    proxy_val = proxy[split:]
    pnl_val = pnl[split:]
    cross_val = cross_col[split:]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 96),
        }
        _, score = train_lgbm_candidate(
            x_train,
            y_train,
            x_val,
            y_val,
            proxy_val,
            pnl_val,
            cross_val,
            params,
        )
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best_params = study.best_params
    model, best_score = train_lgbm_candidate(
        x_train,
        y_train,
        x_val,
        y_val,
        proxy_val,
        pnl_val,
        cross_val,
        best_params,
    )
    bundle_meta = {
        "blend_weight": 0.55,
        "feature_dim": META_FEATURE_DIM,
        "base_feature_dim": FEATURE_DIM,
        "cross_symbol_feature_count": CROSS_SYMBOL_FEATURE_COUNT,
        "flow_feature_count": 2,
        "feature_names": meta_classifier_column_names(),
        **best_params,
    }
    return model, bundle_meta, best_score
