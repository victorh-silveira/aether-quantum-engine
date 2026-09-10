"""Montagem vetorizada do dataset meta-classificador com alinhamento temporal epoch."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

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
INNER_JOIN_MIN_SAMPLE_RATIO = 0.50
TCN_CALL_PROXY_THRESHOLD = 0.55
TCN_PUT_PROXY_THRESHOLD = 0.45
MICRO_ZSCORE_WINDOW = 1024
TARGET_WINSOR_Q_LOW = 0.01
TARGET_WINSOR_Q_HIGH = 0.99
GRAY_KEEP_MIN_RATIO = 0.50
GRAY_KEEP_MIN_ROWS = 40
GRAY_FILTER_HARD = 1
GRAY_FILTER_SOFT = 0
FORWARD_TARGET_ZSCORE_WINDOW = 64
LABEL_MODE_FORWARD_Z = 1
LABEL_MODE_PAYOFF = 2
FWD_TARGET_VAR_FLOOR = 1e-12
Z_COLLAPSE_MAX_RATIO = 0.50
PAYOFF_SCALE_STD_FLOOR = 1e-6
PAYOFF_ABS_FLAT_FLOOR = 1e-12
TARGET_PREFIX_MIN_KEEP = 150
TRAIN_VAL_STD_RATIO_MAX = 1.5
TARGET_NULL_MAE_GAP_MAX = 2.0


def _proxy_prob_from_past_return(past_return: np.ndarray) -> np.ndarray:
    return np.clip(0.5 + 0.15 * past_return, 0.05, 0.95).astype(np.float32)


def _rolling_zscore_strict(values: np.ndarray, *, window: int = MICRO_ZSCORE_WINDOW) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    min_samples = max(8, window // 8)
    series = pl.Series(arr)
    mean = series.rolling_mean(window_size=window, min_samples=min_samples).to_numpy()
    std = series.rolling_std(window_size=window, min_samples=min_samples, ddof=0).to_numpy()
    std_safe = np.where(std == 0.0, np.nan, std)
    return (arr - mean) / std_safe


def _finalize_payoff_labels(
    payoff: np.ndarray,
    meta: dict[str, float | int],
) -> tuple[np.ndarray, dict[str, float | int]]:
    meta["label_mode"] = int(LABEL_MODE_PAYOFF)
    meta["label_scale"] = 1.0
    return np.asarray(payoff, dtype=np.float32), meta


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
        "label_scale": 1.0,
    }
    if fwd_var > FWD_TARGET_VAR_FLOOR and close_nunique >= 8:
        payoff = _continuous_payoff_target(proxy, fwd.astype(np.float32), bear, stake=float(stake))
        if float(np.var(payoff)) > FWD_TARGET_VAR_FLOOR:
            return _finalize_payoff_labels(payoff, meta)
        z_raw = _rolling_zscore_strict(fwd, window=FORWARD_TARGET_ZSCORE_WINDOW)
        collapse = float(np.mean(~np.isfinite(z_raw))) if z_raw.size else 1.0
        meta["z_collapse_pct"] = int(round(100.0 * collapse))
        z_filled = np.nan_to_num(z_raw, nan=0.0, posinf=0.0, neginf=0.0)
        if collapse + 1e-12 < Z_COLLAPSE_MAX_RATIO and float(np.var(z_filled)) > FWD_TARGET_VAR_FLOOR:
            meta["label_mode"] = int(LABEL_MODE_FORWARD_Z)
            meta["label_scale"] = 1.0
            return z_filled.astype(np.float32), meta
    payoff = _continuous_payoff_target(proxy, fwd.astype(np.float32), bear, stake=float(stake))
    return _finalize_payoff_labels(payoff, meta)


def teacher_decisive_mask(proxy: np.ndarray) -> np.ndarray:
    arr = np.asarray(proxy, dtype=np.float64)
    return (arr <= float(TCN_PUT_PROXY_THRESHOLD)) | (arr >= float(TCN_CALL_PROXY_THRESHOLD))


def teacher_sample_weights(proxy: np.ndarray) -> np.ndarray:
    conf = 2.0 * np.abs(np.asarray(proxy, dtype=np.float64) - 0.5)
    return np.clip(conf, 0.1, 1.0).astype(np.float64)


def _target_std(y: np.ndarray) -> float:
    arr = np.asarray(y, dtype=np.float64)
    return float(np.std(arr, ddof=0)) if arr.size else 0.0


def _fit_train_target_transform(y_train: np.ndarray) -> tuple[float, float, float]:
    arr = np.asarray(y_train, dtype=np.float64)
    if arr.size < 8:
        lo = float(np.min(arr)) if arr.size else 0.0
        hi = float(np.max(arr)) if arr.size else 0.0
    else:
        q_lo, q_hi = np.quantile(arr, [TARGET_WINSOR_Q_LOW, TARGET_WINSOR_Q_HIGH])
        lo, hi = float(q_lo), float(q_hi)
        if not np.isfinite(lo) or not np.isfinite(hi):
            lo = float(np.min(arr))
            hi = float(np.max(arr))
    if hi <= lo:
        hi = lo + 1.0
    clipped = np.clip(arr, lo, hi)
    std = float(np.std(clipped, ddof=0)) if clipped.size else 0.0
    return lo, hi, float(max(std, PAYOFF_SCALE_STD_FLOOR))


def _apply_target_transform(y: np.ndarray, lo: float, hi: float, scale: float) -> np.ndarray:
    arr = np.clip(np.asarray(y, dtype=np.float64), float(lo), float(hi))
    return (arr / max(float(scale), PAYOFF_SCALE_STD_FLOOR)).astype(np.float32)


def _scale_targets_from_train(
    y_train: np.ndarray,
    y_val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    lo, hi, scale = _fit_train_target_transform(y_train)
    return _apply_target_transform(y_train, lo, hi, scale), _apply_target_transform(y_val, lo, hi, scale), scale


def _purged_split_arrays(y: np.ndarray, *, embargo: int = 32) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(y, dtype=np.float64)
    sample_count = int(arr.size)
    val_size = max(32, int(sample_count * 0.25))
    train_end = sample_count - val_size - max(0, int(embargo))
    if train_end < 32 or val_size < 8:
        cut = max(1, int(sample_count * 0.8))
        return arr[:cut], arr[cut:]
    val_start = train_end + max(0, int(embargo))
    return arr[:train_end], arr[val_start:]


def _purged_split_stds(y: np.ndarray, *, embargo: int = 32) -> tuple[float, float]:
    y_train, y_val = _purged_split_arrays(y, embargo=embargo)
    return _target_std(y_train), _target_std(y_val)


def _null_mae_gap(y_train: np.ndarray, y_val: np.ndarray) -> float:
    y_tr, y_va, _ = _scale_targets_from_train(y_train, y_val)
    loc = float(np.median(np.asarray(y_tr, dtype=np.float64)))
    train_mae = float(np.mean(np.abs(np.asarray(y_tr, dtype=np.float64) - loc)))
    val_mae = float(np.mean(np.abs(np.asarray(y_va, dtype=np.float64) - loc)))
    return val_mae / max(train_mae, 1e-9)


def _split_scale_is_trainable(y: np.ndarray) -> bool:
    y_train, y_val = _purged_split_arrays(y)
    train_std = _target_std(y_train)
    val_std = _target_std(y_val)
    if train_std < PAYOFF_SCALE_STD_FLOOR:
        return False
    if (val_std / max(train_std, PAYOFF_SCALE_STD_FLOOR)) > TRAIN_VAL_STD_RATIO_MAX + 1e-12:
        return False
    return _null_mae_gap(y_train, y_val) <= TARGET_NULL_MAE_GAP_MAX + 1e-12


def _leading_trainable_offset(y: np.ndarray, *, min_keep: int) -> int:
    arr = np.asarray(y, dtype=np.float64)
    n = int(arr.size)
    keep_floor = min(n, max(1, int(min_keep)))
    start = 0
    limit = max(0, n - keep_floor)
    best_start = 0
    best_gap = float("inf")
    while start <= limit:
        y_train, y_val = _purged_split_arrays(arr[start:])
        gap = _null_mae_gap(y_train, y_val) if y_train.size and y_val.size else float("inf")
        if _split_scale_is_trainable(arr[start:]):
            return int(start)
        if gap < best_gap:
            best_gap = gap
            best_start = start
        start += 8
    return int(best_start)


def _assert_train_val_target_scale(
    train_std: float,
    val_std: float,
    *,
    null_mae_gap: float | None = None,
) -> None:
    train = float(train_std)
    val = float(val_std)
    if train < PAYOFF_SCALE_STD_FLOOR:
        raise RuntimeError(
            "Export meta bloqueado: split de alvo degenerado "
            f"(train_std={train:.6f} val_std={val:.6f}). "
            "Treino sem variancia; nao treinar meta neste historico."
        )
    ratio = val / max(train, PAYOFF_SCALE_STD_FLOOR)
    if ratio > TRAIN_VAL_STD_RATIO_MAX + 1e-12:
        raise RuntimeError(
            "Export meta bloqueado: split de alvo degenerado "
            f"(train_std={train:.6f} val_std={val:.6f} ratio={ratio:.1f} "
            f"> {TRAIN_VAL_STD_RATIO_MAX:.1f}). "
            "Prefixo plano ou lookahead de escala; nao treinar meta neste historico."
        )
    if null_mae_gap is not None and float(null_mae_gap) > TARGET_NULL_MAE_GAP_MAX + 1e-12:
        raise RuntimeError(
            "Export meta bloqueado: split de alvo degenerado "
            f"(null_mae_gap={float(null_mae_gap):.3f} > {TARGET_NULL_MAE_GAP_MAX:.1f}). "
            "Mediana L1 do treino ja fura o teto MAE; nao treinar meta neste historico."
        )


def _slice_training_rows(
    frame: pl.DataFrame,
    labels: np.ndarray,
    proxy: np.ndarray,
    call_pnl: np.ndarray,
    start: int,
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    n = int(np.asarray(labels).size)
    if start <= 0:
        return frame, np.asarray(labels), np.asarray(proxy), np.asarray(call_pnl)
    return (
        frame.slice(start, n - start),
        np.asarray(labels[start:], dtype=np.float32),
        np.asarray(proxy[start:], dtype=np.float32),
        np.asarray(call_pnl[start:], dtype=np.float32),
    )


def _trim_degenerate_target_prefix(
    frame: pl.DataFrame,
    labels: np.ndarray,
    proxy: np.ndarray,
    call_pnl: np.ndarray,
    *,
    min_keep: int = TARGET_PREFIX_MIN_KEEP,
    abs_floor: float = PAYOFF_ABS_FLAT_FLOOR,
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray, np.ndarray, int]:
    y = np.asarray(labels, dtype=np.float64)
    n = int(y.size)
    keep_floor = min(n, max(1, int(min_keep)))
    start = 0
    limit = n - keep_floor
    while start < limit and abs(float(y[start])) <= float(abs_floor):
        start += 1
    rest = y[start:]
    start += _leading_trainable_offset(rest, min_keep=min_keep)
    if start <= 0:
        return frame, np.asarray(labels), np.asarray(proxy), np.asarray(call_pnl), 0
    sliced = _slice_training_rows(frame, labels, proxy, call_pnl, start)
    return sliced[0], sliced[1], sliced[2], sliced[3], int(start)


def _forward_return_z_target(
    forward: np.ndarray,
    *,
    window: int = FORWARD_TARGET_ZSCORE_WINDOW,
) -> np.ndarray:
    return _rolling_zscore(np.asarray(forward, dtype=np.float64), window=int(window)).astype(np.float32)


def _rolling_zscore(values: np.ndarray, *, window: int = MICRO_ZSCORE_WINDOW) -> np.ndarray:
    z = _rolling_zscore_strict(values, window=window)
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


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
    delta = pl.Series(np.asarray(closes, dtype=np.float64)).diff()
    tick_accel = delta.diff().rolling_mean(window_size=5, min_samples=2).fill_null(0.0).to_numpy()
    return np.asarray(tick_accel, dtype=np.float64), keltner_dev


def _micro_vol_arrays(
    closes: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    series: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    close_s = pl.Series(np.asarray(closes, dtype=np.float64))
    momentum = close_s.diff().rolling_mean(window_size=5, min_samples=2).fill_null(0.0).to_numpy()
    momentum = np.asarray(momentum, dtype=np.float64)
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
) -> tuple[pl.DataFrame, np.ndarray]:
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
    tick_count = (
        np.asarray(series["tick_count"], dtype=np.float64)
        if len(series.get("tick_count", [])) > 0
        else np.zeros(len(closes), dtype=np.float64)
    )
    price_vel = (
        np.asarray(series["price_velocity"], dtype=np.float64)
        if len(series.get("price_velocity", [])) > 0
        else np.zeros(len(closes), dtype=np.float64)
    )
    implied = (
        np.asarray(series["implied_vol_ratio"], dtype=np.float64)
        if len(series.get("implied_vol_ratio", [])) > 0
        else np.ones(len(closes), dtype=np.float64)
    )
    frame = pl.DataFrame(
        {
            "epoch": bundle.epochs.astype(np.int64),
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
            "micro_price_velocity": np.clip(price_vel, -3.0, 3.0),
            "micro_tick_count_norm": np.clip(tick_count / 300.0, 0.0, 1.0),
            "implied_vol_centered": np.clip(implied - 1.0, -3.0, 3.0),
        }
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
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    planned_fetch = int(fetch_count)
    if not bundles:
        raise RuntimeError("Treino meta-classificador exige ao menos um bundle OHLC.")
    by_symbol = {bundle.symbol: bundle for bundle in bundles}
    primary = by_symbol.get("1HZ75V") or next(iter(by_symbol.values()))
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
    sorted_df = df_primary.sort("epoch")
    paired_cap = int(sorted_df.height)
    rows = paired_cap - FEATURE_LOOKBACK_SKIP - max(2, label_horizon)
    if rows <= 0:
        raise RuntimeError("Historico insuficiente para montar features meta single-symbol.")
    effective_fetch = min(planned_fetch, paired_cap)
    _validate_sample_floor(rows, effective_fetch)
    start = FEATURE_LOOKBACK_SKIP
    primary_slice = sorted_df.slice(start, rows)
    orig_epochs = df_primary["epoch"].to_numpy()
    epoch_to_row = {int(epoch): row for row, epoch in enumerate(orig_epochs)}
    epoch_slice = primary_slice["epoch"].to_numpy()
    row_idx = np.asarray([epoch_to_row[int(epoch)] for epoch in epoch_slice], dtype=np.int64)
    base_features = primary_features[row_idx]
    flow_tick = primary_slice["tick_accel"].to_numpy().astype(np.float64)
    flow_keltner = primary_slice["keltner_dev"].to_numpy().astype(np.float64)
    matrix = np.column_stack(
        [
            base_features,
            primary_slice["micro_bid_ask_spread_momentum"].to_numpy().astype(np.float32),
            primary_slice["micro_bid_ask_spread_momentum_zscore"].to_numpy().astype(np.float32),
            primary_slice["volatility_shadow_ratio"].to_numpy().astype(np.float32),
            primary_slice["volatility_shadow_ratio_zscore"].to_numpy().astype(np.float32),
            primary_slice["micro_price_velocity"].to_numpy().astype(np.float32),
            primary_slice["micro_tick_count_norm"].to_numpy().astype(np.float32),
            primary_slice["implied_vol_centered"].to_numpy().astype(np.float32),
            flow_tick,
            flow_keltner,
        ]
    ).astype(np.float32)
    if matrix.shape[1] != META_FEATURE_DIM:
        raise RuntimeError(f"Meta feature row divergente: esperado {META_FEATURE_DIM}, obtido {matrix.shape[1]}")
    proxy = primary_slice["prob_call"].to_numpy().astype(np.float32)
    call_pnl = primary_slice["pnl"].to_numpy().astype(np.float32)
    close_slice = np.asarray(primary.closes, dtype=np.float64)[row_idx]
    labels, label_meta = _resolve_training_labels(
        proxy,
        call_pnl,
        closes=close_slice,
        stake=float(reference_stake),
    )
    columns = meta_classifier_column_names()
    frame = pl.DataFrame({name: matrix[:, idx] for idx, name in enumerate(columns)})
    frame, labels, proxy, call_pnl, n_dropped_flat_prefix = _trim_degenerate_target_prefix(
        frame,
        labels,
        proxy,
        call_pnl,
    )
    n_kept = int(len(labels))
    hygiene = {
        "n_before_gray_filter": n_kept,
        "n_dropped_gray": 0,
        "n_gray_soft_retained": 0,
        "n_dropped_flat_prefix": int(n_dropped_flat_prefix),
        "n_kept": n_kept,
        "gray_filter_mode": int(GRAY_FILTER_SOFT),
        "label_mode": int(label_meta["label_mode"]),
        "label_scale": float(label_meta.get("label_scale", 1.0)),
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
