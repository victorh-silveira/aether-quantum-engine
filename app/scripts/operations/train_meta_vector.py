"""Montagem vetorizada do dataset meta-classificador com alinhamento temporal epoch."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

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


def _proxy_prob_from_past_return(past_return: np.ndarray) -> np.ndarray:
    return np.clip(0.5 + 0.15 * past_return, 0.05, 0.95).astype(np.float32)


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


def _inner_join_symbol_frames(
    df_bull: pd.DataFrame,
    df_bear: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Index]:
    bull_reset = df_bull.reset_index()
    bear_reset = df_bear.reset_index()
    merged = pd.merge(bull_reset, bear_reset, on="epoch", how="inner", suffixes=("_bull", "_bear"))
    if merged.empty:
        raise RuntimeError(
            "Nenhuma epoch comum entre RDBULL e RDBEAR apos inner join. "
            "Reinicie o treino meta-regressor para forcar nova paginacao limpa via WebSocket/REST."
        )
    ordered_epochs = pd.Index(merged["epoch"].astype(np.int64)).sort_values()
    bull_cols = list(df_bull.columns)
    bear_cols = list(df_bear.columns)
    bull_aligned = merged.set_index("epoch").loc[ordered_epochs, [f"{column}_bull" for column in bull_cols]]
    bull_aligned.columns = bull_cols
    bear_aligned = merged.set_index("epoch").loc[ordered_epochs, [f"{column}_bear" for column in bear_cols]]
    bear_aligned.columns = bear_cols
    return bull_aligned, bear_aligned, ordered_epochs


def _bull_feature_rows_for_epochs(
    df_bull: pd.DataFrame,
    bull_features: np.ndarray,
    epochs: pd.Index,
) -> np.ndarray:
    epoch_to_row = {int(epoch): row for row, epoch in enumerate(df_bull.index.astype(np.int64))}
    row_idx = np.asarray([epoch_to_row[int(epoch)] for epoch in epochs], dtype=np.int64)
    return bull_features[row_idx]


def _validate_inner_join_sample_floor(rows: int, fetch_count: int) -> None:
    minimum = int(max(1, fetch_count) * INNER_JOIN_MIN_SAMPLE_RATIO)
    if rows >= minimum:
        return
    raise RuntimeError(
        "Alinhamento epoch inner join insuficiente apos paginacao assimétrica: "
        f"{rows} amostras pareadas (minimo {minimum} para fetch_count={fetch_count}). "
        "Reinicie o comando de treino meta-regressor para forcar nova paginacao limpa via WebSocket/REST."
    )


def build_paired_training_dataset(
    bundles: list[OhlcBundle],
    *,
    micro_granularity: int = 120,
    contract_duration_seconds: int | None = None,
    reference_stake: float = META_TRAIN_REFERENCE_STAKE,
    fetch_count: int = META_TRAIN_DEFAULT_BARS,
    teacher_probs: dict[str, np.ndarray] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    planned_fetch = int(fetch_count)
    by_symbol = {bundle.symbol: bundle for bundle in bundles}
    bull = by_symbol.get("RDBULL")
    bear = by_symbol.get("RDBEAR")
    if bull is None or bear is None:
        raise RuntimeError("Treino meta-classificador exige bundles RDBULL e RDBEAR alinhados.")
    duration = int(contract_duration_seconds) if contract_duration_seconds is not None else int(micro_granularity)
    label_horizon = _resolve_label_horizon_bars(
        int(bull.granularity),
        micro_granularity=int(micro_granularity),
        contract_duration_seconds=duration,
    )
    df_bull, bull_features = _symbol_frame(
        bull,
        label_horizon_bars=label_horizon,
        teacher_probs=teacher_probs,
    )
    df_bear, _bear_features = _symbol_frame(
        bear,
        label_horizon_bars=label_horizon,
        teacher_probs=teacher_probs,
    )
    bull_aligned, bear_aligned, epoch_index = _inner_join_symbol_frames(df_bull, df_bear)
    paired_cap = min(len(bull_aligned), len(bear_aligned))
    rows = len(epoch_index) - FEATURE_LOOKBACK_SKIP - max(2, label_horizon)
    if rows <= 0:
        raise RuntimeError("Historico insuficiente para parear features cross-symbol.")
    effective_fetch = min(planned_fetch, paired_cap)
    _validate_inner_join_sample_floor(rows, effective_fetch)
    start = FEATURE_LOOKBACK_SKIP
    end = start + rows
    epoch_slice = epoch_index[start:end]
    bull_slice = bull_aligned.loc[epoch_slice]
    bear_slice = bear_aligned.loc[epoch_slice]
    base_features = _bull_feature_rows_for_epochs(df_bull, bull_features, epoch_slice)
    bull_call = bull_slice["prob_call"].to_numpy(dtype=np.float64)
    bear_put = 1.0 - bear_slice["prob_call"].to_numpy(dtype=np.float64)
    cross_prob_delta = np.abs(bull_call - bear_put)
    cross_vol_diff = bull_slice["vol_ratio"].to_numpy(dtype=np.float64) - bear_slice["vol_ratio"].to_numpy(
        dtype=np.float64
    )
    cross_rsi_spread = bull_slice["rsi"].to_numpy(dtype=np.float64) - bear_slice["rsi"].to_numpy(dtype=np.float64)
    flow_tick = bull_slice["tick_accel"].to_numpy(dtype=np.float64)
    flow_keltner = bull_slice["keltner_dev"].to_numpy(dtype=np.float64)
    matrix = np.column_stack(
        [
            base_features,
            bull_slice["micro_bid_ask_spread_momentum"].to_numpy(dtype=np.float32),
            bull_slice["micro_bid_ask_spread_momentum_zscore"].to_numpy(dtype=np.float32),
            bull_slice["volatility_shadow_ratio"].to_numpy(dtype=np.float32),
            bull_slice["volatility_shadow_ratio_zscore"].to_numpy(dtype=np.float32),
            cross_prob_delta,
            cross_vol_diff,
            cross_rsi_spread,
            flow_tick,
            flow_keltner,
        ]
    ).astype(np.float32)
    if matrix.shape[1] != META_FEATURE_DIM:
        raise RuntimeError(f"Meta feature row divergente: esperado {META_FEATURE_DIM}, obtido {matrix.shape[1]}")
    proxy = bull_slice["prob_call"].to_numpy(dtype=np.float32)
    bull_pnl = bull_slice["pnl"].to_numpy(dtype=np.float32)
    bear_pnl = bear_slice["pnl"].to_numpy(dtype=np.float32)
    labels = _continuous_payoff_target(proxy, bull_pnl, bear_pnl, stake=reference_stake)
    frame = pd.DataFrame(matrix, columns=meta_classifier_column_names())
    return frame, labels, proxy, bull_pnl


def load_teacher_probs_from_checkpoints(
    symbols: list[str],
    *,
    model_path_template: str,
    lookback: int,
    repo_root: Path,
) -> dict[str, np.ndarray]:
    _ = (lookback, repo_root)
    loaded: dict[str, np.ndarray] = {}
    for symbol in symbols:
        path = Path(str(model_path_template).format(symbol=symbol))
        if not path.is_file():
            continue
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            logging.getLogger("META_TRAIN").debug("teacher checkpoint skip %s: %s", path, exc)
            continue
        probs = payload.get("teacher_probs") if isinstance(payload, dict) else None
        if probs is None and isinstance(payload, dict):
            probs = payload.get("val_probs")
        if probs is None:
            continue
        arr = np.asarray(probs, dtype=np.float32).reshape(-1)
        if arr.size > 0:
            loaded[str(symbol)] = arr
    return loaded


def resolve_contract_duration_seconds(settings: dict[str, Any]) -> int:
    risk = settings.get("risk_management") if isinstance(settings.get("risk_management"), dict) else {}
    params = risk.get("params") if isinstance(risk, dict) else {}
    if isinstance(params, dict) and params.get("duration") is not None:
        return max(1, int(params["duration"]))
    data = settings.get("data_handler") if isinstance(settings.get("data_handler"), dict) else {}
    return max(1, int(data.get("micro_granularity", 120))) if isinstance(data, dict) else 120
