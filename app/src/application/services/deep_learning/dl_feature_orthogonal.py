"""Montagem do vetor TCN ortogonal 14D a partir das series precomputadas."""

from __future__ import annotations

import numpy as np

from src.application.services.deep_learning.dl_feature_normalize import (
    apply_causal_column_scale,
    center_unit_interval,
)


FEATURE_DIM = 14
ORTHOGONAL_FEATURE_NAMES: tuple[str, ...] = (
    "norm_log_ret_1",
    "norm_log_ret_5",
    "rsi_centered",
    "delta_rsi",
    "bb_pct_b_centered",
    "bb_log_width",
    "norm_atr",
    "ema_dist_9_21",
    "ema_dist_20_50",
    "macd_hist_norm",
    "stoch_k_centered",
    "adx_scaled",
    "realized_vol_ratio",
    "hurst_centered",
)
UNBOUNDED_COLS: tuple[int, ...] = (0, 1, 3, 5, 6, 7, 8, 9, 12, 13)


def _log_ret_n(log_return: np.ndarray, n: int) -> np.ndarray:
    """Soma causal de log-retornos nas ultimas n barras."""
    out = np.zeros(len(log_return), dtype=np.float64)
    for i in range(len(log_return)):
        start = max(0, i - n + 1)
        out[i] = float(np.sum(log_return[start : i + 1]))
    return out


def _hurst_centered(series: dict[str, np.ndarray], n: int) -> np.ndarray:
    """Hurst centrado em 0.5 (persistencia vs mean-reversion)."""
    hurst = np.asarray(series.get("hurst", np.full(n, 0.5)), dtype=np.float64)
    if len(hurst) != n:
        out = np.full(n, 0.0, dtype=np.float64)
        take = min(len(hurst), n)
        out[:take] = hurst[:take] - 0.5
        return out
    return hurst - 0.5


def build_orthogonal_raw_matrix(
    series: dict[str, np.ndarray],
    *,
    macro_closes: np.ndarray | None = None,
) -> np.ndarray:
    """Monta matriz (N, 14) bruta antes da escala causal unbounded."""
    del macro_closes
    n = len(series["log_return"])
    log_return = np.asarray(series["log_return"], dtype=np.float64)
    rsi = np.asarray(series["rsi"], dtype=np.float64)
    bb_pct = np.asarray(series["bb_pct_b"], dtype=np.float64)
    bb_w = np.asarray(series.get("bb_width_raw", series["bb_width"]), dtype=np.float64)
    atr = np.asarray(series.get("atr_raw", series["atr_norm"]), dtype=np.float64)
    macd = np.asarray(series["macd"], dtype=np.float64)
    macd_sig = np.asarray(series["macd_signal"], dtype=np.float64)
    stoch = np.asarray(series["stoch_k"], dtype=np.float64)
    adx = np.asarray(series["adx"], dtype=np.float64)
    cols = [
        log_return,
        _log_ret_n(log_return, 5),
        center_unit_interval(rsi),
        np.asarray(series["delta_rsi"], dtype=np.float64),
        center_unit_interval(bb_pct),
        np.log(np.maximum(bb_w, 1e-10)),
        atr,
        np.asarray(series["ema_9_21_dist"], dtype=np.float64),
        np.asarray(series.get("ema_20_50_dist", series["ema_dist_50"]), dtype=np.float64),
        macd - macd_sig,
        center_unit_interval(stoch),
        np.clip(adx, 0.0, 1.0),
        np.asarray(series["vol_ratio_short_long"], dtype=np.float64),
        _hurst_centered(series, n),
    ]
    return np.stack(cols, axis=1).astype(np.float32)


def build_orthogonal_feature_matrix(
    series: dict[str, np.ndarray],
    *,
    macro_closes: np.ndarray | None = None,
    causal_norm_window: int = 288,
    causal_norm_clip: float = 3.0,
) -> np.ndarray:
    """Matriz (N, 14) com osciladores em [-1,1] e unbounded via Median/IQR causal."""
    raw = build_orthogonal_raw_matrix(series, macro_closes=macro_closes)
    return apply_causal_column_scale(
        raw,
        UNBOUNDED_COLS,
        window=causal_norm_window,
        clip=causal_norm_clip,
    )


def build_orthogonal_feature_row(
    series: dict[str, np.ndarray],
    index: int,
    *,
    macro_closes: np.ndarray | None = None,
    causal_norm_window: int = 288,
    causal_norm_clip: float = 3.0,
) -> np.ndarray:
    """Linha 14D na barra index (recomputa matriz ate index para causalidade)."""
    matrix = build_orthogonal_feature_matrix(
        series,
        macro_closes=macro_closes,
        causal_norm_window=causal_norm_window,
        causal_norm_clip=causal_norm_clip,
    )
    return matrix[index].astype(np.float32)
