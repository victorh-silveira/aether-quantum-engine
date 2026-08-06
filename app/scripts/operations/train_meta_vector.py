"""Montagem vetorizada do dataset meta-classificador com alinhamento temporal epoch."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from scripts.operations.train_meta_data import META_TRAIN_DEFAULT_BARS, OhlcBundle
from src.application.services.deep_learning.dl_feature_build import precompute_price_series
from src.application.services.deep_learning.dl_feature_matrix import build_feature_matrix
from src.application.services.meta_classifier_cross_symbol import META_FEATURE_DIM
from src.application.services.meta_classifier_features import (
    clip_feature_zscore,
    meta_classifier_column_names,
)


FEATURE_LOOKBACK_SKIP = 32
META_TRAIN_REFERENCE_STAKE = 1.0
INNER_JOIN_MIN_SAMPLE_RATIO = 0.80
TCN_CALL_PROXY_THRESHOLD = 0.53
TCN_PUT_PROXY_THRESHOLD = 0.47
MICRO_ZSCORE_WINDOW = 1024
TARGET_WINSOR_Q_LOW = 0.01
TARGET_WINSOR_Q_HIGH = 0.99
GRAY_KEEP_MIN_RATIO = 0.50
GRAY_KEEP_MIN_ROWS = 96
GRAY_FILTER_HARD = 1
GRAY_FILTER_SOFT = 0
FORWARD_TARGET_ZSCORE_WINDOW = 64
LABEL_MODE_FORWARD_Z = 1
LABEL_MODE_PAYOFF = 2
FWD_TARGET_VAR_FLOOR = 1e-12
Z_COLLAPSE_MAX_RATIO = 0.50


def _proxy_prob_from_past_return(past_return: np.ndarray) -> np.ndarray:
    return np.clip(0.5 + 0.15 * past_return, 0.05, 0.95).astype(np.float32)


def _rolling_zscore_strict(values: np.ndarray, *, window: int = MICRO_ZSCORE_WINDOW) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=np.float64))
    mean = series.rolling(window, min_periods=max(8, window // 8)).mean()
    std = series.rolling(window, min_periods=max(8, window // 8)).std(ddof=0)
    z = (series - mean) / std.replace(0.0, np.nan)
    return z.to_numpy(dtype=np.float64)


def _resolve_training_labels(
    proxy: np.ndarray,
    call_pnl: np.ndarray,
    *,
    closes: np.ndarray,
    stake: float = META_TRAIN_REFERENCE_STAKE,
) -> tuple[np.ndarray, dict[str, float | int]]:
    fwd = np.asarray(call_pnl, dtype=np.float64)
    fwd_var = float(np.var(fwd)) if fwd.size else 0.0
    close_nunique = int(len(np.unique(np.round(np.asarray(closes, dtype=np.float64), decimals=8))))
    bear = (-fwd).astype(np.float32)
    meta: dict[str, float | int] = {
        "forward_var": fwd_var,
        "close_nunique": close_nunique,
        "z_collapse_pct": 0,
        "label_mode": int(LABEL_MODE_FORWARD_Z),
    }
    if fwd_var > FWD_TARGET_VAR_FLOOR and close_nunique >= 8:
        z_raw = _rolling_zscore_strict(fwd, window=FORWARD_TARGET_ZSCORE_WINDOW)
        collapse = float(np.mean(~np.isfinite(z_raw))) if z_raw.size else 1.0
        meta["z_collapse_pct"] = int(round(100.0 * collapse))
        z_filled = np.nan_to_num(z_raw, nan=0.0, posinf=0.0, neginf=0.0)
        if collapse + 1e-12 < Z_COLLAPSE_MAX_RATIO and float(np.var(z_filled)) > FWD_TARGET_VAR_FLOOR:
            labels = _winsorize_target(z_filled.astype(np.float32))
            meta["label_mode"] = int(LABEL_MODE_FORWARD_Z)
            return labels, meta
    labels = _winsorize_target(_continuous_payoff_target(proxy, fwd.astype(np.float32), bear, stake=float(stake)))
    meta["label_mode"] = int(LABEL_MODE_PAYOFF)
    return labels, meta


def teacher_decisive_mask(proxy: np.ndarray) -> np.ndarray:
    arr = np.asarray(proxy, dtype=np.float64)
    return (arr <= float(TCN_PUT_PROXY_THRESHOLD)) | (arr >= float(TCN_CALL_PROXY_THRESHOLD))


def teacher_sample_weights(proxy: np.ndarray) -> np.ndarray:
    conf = 2.0 * np.abs(np.asarray(proxy, dtype=np.float64) - 0.5)
    return np.clip(conf, 0.1, 1.0).astype(np.float64)


def _winsorize_target(y: np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float64)
    if arr.size < 8:
        return arr.astype(np.float32)
    lo, hi = np.quantile(arr, [TARGET_WINSOR_Q_LOW, TARGET_WINSOR_Q_HIGH])
    if not np.isfinite(lo) or not np.isfinite(hi) or float(hi) <= float(lo):
        return arr.astype(np.float32)
    return np.clip(arr, float(lo), float(hi)).astype(np.float32)


def _forward_return_z_target(
    forward: np.ndarray,
    *,
    window: int = FORWARD_TARGET_ZSCORE_WINDOW,
) -> np.ndarray:
    return _rolling_zscore(np.asarray(forward, dtype=np.float64), window=int(window)).astype(np.float32)


def _rolling_zscore(values: np.ndarray, *, window: int = MICRO_ZSCORE_WINDOW) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=np.float64))
    mean = series.rolling(window, min_periods=max(8, window // 8)).mean()
    std = series.rolling(window, min_periods=max(8, window // 8)).std(ddof=0)
    z = (series - mean) / std.replace(0.0, np.nan)
    return np.nan_to_num(z.to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def _continuous_payoff_target(
    proxy: np.ndarray,
    bull_forward: np.ndarray,
    bear_forward: np.ndarray,
    *,
    stake: float = META_TRAIN_REFERENCE_STAKE,
) -> np.ndarray:
    call_mask = proxy >= TCN_CALL_PROXY_THRESHOLD
    put_mask = proxy <= TCN_PUT_PROXY_THRESHOLD
    gray_bull = proxy > 0.5
    signed_pnl = np.where(
        call_mask,
        bull_forward,
        np.where(put_mask, bear_forward, np.where(gray_bull, bull_forward, bear_forward)),
    )
    return (signed_pnl / max(float(stake), 1e-9)).astype(np.float32)


def _flow_arrays(closes: np.ndarray, series: dict) -> tuple[np.ndarray, np.ndarray]:
    keltner_pct = series.get("keltner_pct_b")
    if keltner_pct is None or len(keltner_pct) == 0:
        keltner_dev = np.zeros(len(closes), dtype=np.float64)
    else:
        keltner_dev = np.asarray(keltner_pct, dtype=np.float64) - 0.5
    delta = pd.Series(closes, dtype=np.float64).diff()
    tick_accel = delta.diff().rolling(5, min_periods=2).mean().fillna(0.0).to_numpy(dtype=np.float64)
    return tick_accel, keltner_dev


def _micro_vol_arrays(
    closes: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    series: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    close_s = pd.Series(closes, dtype=np.float64)
    momentum = close_s.diff().rolling(5, min_periods=2).mean().fillna(0.0).to_numpy(dtype=np.float64)
    momentum_z = np.asarray(
        [clip_feature_zscore(float(v)) for v in _rolling_zscore(momentum)],
        dtype=np.float64,
    )
    atr = series.get("atr_norm")
    if atr is None or len(atr) == 0:
        atr_arr = np.maximum(np.asarray(high, dtype=np.float64) - np.asarray(low, dtype=np.float64), 1e-9)
    else:
        atr_arr = np.maximum(np.abs(np.asarray(atr, dtype=np.float64)) + 1e-6, 1e-9)
    upper = np.asarray(high, dtype=np.float64) - np.asarray(closes, dtype=np.float64)
    lower = np.asarray(closes, dtype=np.float64) - np.asarray(low, dtype=np.float64)
    shadow = (np.maximum(upper, 0.0) + np.maximum(lower, 0.0)) / atr_arr
    shadow_z = np.asarray(
        [clip_feature_zscore(float(v)) for v in _rolling_zscore(shadow)],
        dtype=np.float64,
    )
    return (
        momentum.astype(np.float64),
        momentum_z,
        shadow.astype(np.float64),
        shadow_z,
    )


def _forward_return(closes: np.ndarray, *, horizon_bars: int) -> np.ndarray:
    horizon = max(1, int(horizon_bars))
    out = np.zeros(len(closes), dtype=np.float32)
    if len(closes) <= horizon:
        return out
    out[:-horizon] = (closes[horizon:] - closes[:-horizon]).astype(np.float32)
    return out


def _resolve_label_horizon_bars(
    bundle_granularity: int,
    *,
    micro_granularity: int,
    contract_duration_seconds: int,
) -> int:
    gran = max(1, int(bundle_granularity))
    duration = max(1, int(contract_duration_seconds))
    micro = max(1, int(micro_granularity))
    if gran == micro:
        return max(1, int(round(duration / float(micro))))
    return max(1, int(round(duration / float(gran))))


def _load_teacher_probs(symbol: str, n: int, teacher_probs: dict[str, np.ndarray] | None) -> np.ndarray | None:
    if not isinstance(teacher_probs, dict):
        return None
    raw = teacher_probs.get(str(symbol))
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    if len(arr) < n:
        return None
    return arr[-n:].astype(np.float32)


def _symbol_frame(
    bundle: OhlcBundle,
    *,
    label_horizon_bars: int,
    teacher_probs: dict[str, np.ndarray] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    series = precompute_price_series(
        bundle.closes,
        granularity=bundle.granularity,
        symbol=bundle.symbol,
        open_=bundle.open_,
        high=bundle.high,
        low=bundle.low,
    )
    features = build_feature_matrix(series)
    closes = bundle.closes.astype(np.float64)
    forward = _forward_return(closes, horizon_bars=label_horizon_bars)
    past = np.zeros(len(closes), dtype=np.float32)
    past[1:] = (closes[1:] - closes[:-1]).astype(np.float32)
    teacher = _load_teacher_probs(bundle.symbol, len(closes), teacher_probs)
    proxy = teacher if teacher is not None else _proxy_prob_from_past_return(past)
    pnl = forward.copy()
    rsi = (
        np.asarray(series["rsi"], dtype=np.float64)
        if len(series.get("rsi", [])) > 0
        else np.zeros(len(closes), dtype=np.float64)
    )
    vol_ratio = (
        np.asarray(series["vol_ratio_short_long"], dtype=np.float64)
        if len(series.get("vol_ratio_short_long", [])) > 0
        else np.zeros(len(closes), dtype=np.float64)
    )
    tick_accel, keltner_dev = _flow_arrays(closes, series)
    mom, mom_z, shadow, shadow_z = _micro_vol_arrays(closes, bundle.high, bundle.low, series)
    frame = pd.DataFrame(
        {
            "prob_call": proxy.astype(np.float64),
            "pnl": pnl.astype(np.float64),
            "rsi": rsi,
            "vol_ratio": vol_ratio,
            "tick_accel": tick_accel,
            "keltner_dev": keltner_dev,
            "micro_bid_ask_spread_momentum": mom,
            "micro_bid_ask_spread_momentum_zscore": mom_z,
            "volatility_shadow_ratio": shadow,
            "volatility_shadow_ratio_zscore": shadow_z,
        },
        index=pd.Index(bundle.epochs.astype(np.int64), name="epoch"),
    )
    return frame, features


def _validate_sample_floor(rows: int, fetch_count: int) -> None:
    minimum = int(max(1, fetch_count) * INNER_JOIN_MIN_SAMPLE_RATIO)
    if rows >= minimum:
        return
    raise RuntimeError(
        "Historico insuficiente apos montagem single-symbol: "
        f"{rows} amostras (minimo {minimum} para fetch_count={fetch_count}). "
        "Reinicie o comando de treino meta-regressor para forcar nova paginacao limpa via WebSocket/REST."
    )


def build_paired_training_dataset(
    bundles: list[OhlcBundle],
    *,
    micro_granularity: int = 60,
    contract_duration_seconds: int | None = None,
    reference_stake: float = META_TRAIN_REFERENCE_STAKE,
    fetch_count: int = META_TRAIN_DEFAULT_BARS,
    teacher_probs: dict[str, np.ndarray] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    planned_fetch = int(fetch_count)
    if not bundles:
        raise RuntimeError("Treino meta-classificador exige ao menos um bundle OHLC.")
    by_symbol = {bundle.symbol: bundle for bundle in bundles}
    primary = by_symbol.get("OTC_SPC") or next(iter(by_symbol.values()))
    duration = int(contract_duration_seconds) if contract_duration_seconds is not None else int(micro_granularity)
    label_horizon = _resolve_label_horizon_bars(
        int(primary.granularity),
        micro_granularity=int(micro_granularity),
        contract_duration_seconds=duration,
    )
    df_primary, primary_features = _symbol_frame(
        primary,
        label_horizon_bars=label_horizon,
        teacher_probs=teacher_probs,
    )
    epoch_index = df_primary.index.sort_values()
    paired_cap = len(epoch_index)
    rows = paired_cap - FEATURE_LOOKBACK_SKIP - max(2, label_horizon)
    if rows <= 0:
        raise RuntimeError("Historico insuficiente para montar features meta single-symbol.")
    effective_fetch = min(planned_fetch, paired_cap)
    _validate_sample_floor(rows, effective_fetch)
    start = FEATURE_LOOKBACK_SKIP
    end = start + rows
    epoch_slice = epoch_index[start:end]
    primary_slice = df_primary.loc[epoch_slice]
    epoch_to_row = {int(epoch): row for row, epoch in enumerate(df_primary.index.astype(np.int64))}
    row_idx = np.asarray([epoch_to_row[int(epoch)] for epoch in epoch_slice], dtype=np.int64)
    base_features = primary_features[row_idx]
    zeros = np.zeros(len(epoch_slice), dtype=np.float64)
    flow_tick = primary_slice["tick_accel"].to_numpy(dtype=np.float64)
    flow_keltner = primary_slice["keltner_dev"].to_numpy(dtype=np.float64)
    matrix = np.column_stack(
        [
            base_features,
            primary_slice["micro_bid_ask_spread_momentum"].to_numpy(dtype=np.float32),
            primary_slice["micro_bid_ask_spread_momentum_zscore"].to_numpy(dtype=np.float32),
            primary_slice["volatility_shadow_ratio"].to_numpy(dtype=np.float32),
            primary_slice["volatility_shadow_ratio_zscore"].to_numpy(dtype=np.float32),
            zeros,
            zeros,
            zeros,
            flow_tick,
            flow_keltner,
        ]
    ).astype(np.float32)
    if matrix.shape[1] != META_FEATURE_DIM:
        raise RuntimeError(f"Meta feature row divergente: esperado {META_FEATURE_DIM}, obtido {matrix.shape[1]}")
    proxy = primary_slice["prob_call"].to_numpy(dtype=np.float32)
    call_pnl = primary_slice["pnl"].to_numpy(dtype=np.float32)
    close_slice = np.asarray(primary.closes, dtype=np.float64)[row_idx]
    labels, label_meta = _resolve_training_labels(
        proxy,
        call_pnl,
        closes=close_slice,
        stake=float(reference_stake),
    )
    n_kept = int(len(labels))
    frame = pd.DataFrame(matrix, columns=meta_classifier_column_names())
    hygiene = {
        "n_before_gray_filter": n_kept,
        "n_dropped_gray": 0,
        "n_gray_soft_retained": 0,
        "n_kept": n_kept,
        "gray_filter_mode": int(GRAY_FILTER_SOFT),
        "label_mode": int(label_meta["label_mode"]),
        "forward_var": float(label_meta["forward_var"]),
        "close_nunique": int(label_meta["close_nunique"]),
        "z_collapse_pct": int(label_meta["z_collapse_pct"]),
        "data_source": str(primary.source),
        "bars_loaded": int(len(primary.closes)),
    }
    return frame, labels, proxy, call_pnl, hygiene


def resolve_contract_duration_seconds(settings: dict[str, Any]) -> int:
    risk = settings.get("risk_management") if isinstance(settings.get("risk_management"), dict) else {}
    params = risk.get("params") if isinstance(risk, dict) else {}
    if isinstance(params, dict) and params.get("duration") is not None:
        return max(1, int(params["duration"]))
    data = settings.get("data_handler") if isinstance(settings.get("data_handler"), dict) else {}
    return max(1, int(data.get("micro_granularity", 60))) if isinstance(data, dict) else 60
