"""Montagem de linhas, matrizes e tensores de features DL."""

from __future__ import annotations

import numpy as np

from src.application.services.deep_learning.dl_feature_build import precompute_price_series


def build_feature_row(series: dict[str, np.ndarray], index: int) -> np.ndarray:
    """Monta vetor de features na barra indicada."""
    micro = np.array(
        [
            series["tick_count"][index],
            series["mean_inter_tick_ms"][index] / 1000.0,
            series["price_velocity"][index],
            series["price_acceleration"][index],
            series["consecutive_diff_std"][index],
        ],
        dtype=np.float32,
    )
    traditional = np.array(
        [
            series["rsi"][index],
            series["delta_rsi"][index],
            series["bb_pct_b"][index],
            series["bb_width"][index],
            series["atr_norm"][index],
            series["ema_dist_20"][index],
            series["ema_dist_50"][index],
            series["log_return"][index],
            series["roc"][index],
            series["price_zscore"][index],
            series["macd"][index],
            series["macd_signal"][index],
            series["stoch_k"][index],
            series["stoch_d"][index],
            series["cci"][index],
            series["adx"][index],
            series["di_diff"][index],
            series["williams_r"][index],
            series["ema_9_21_dist"][index],
            series["roc_rsi"][index],
            series["cmo"][index],
            series["keltner_pct_b"][index],
        ],
        dtype=np.float32,
    )
    volatility = np.array(
        [
            series["vol"][index],
            series["vol_vs_target"][index],
            series["vol_z"][index],
            series["implied_vol_ratio"][index],
            series["vol_ratio_short_long"][index],
        ],
        dtype=np.float32,
    )
    persistence = np.array(
        [
            series["hurst"][index],
            series["variance_ratio"][index],
        ],
        dtype=np.float32,
    )
    return np.concatenate([micro, traditional, volatility, persistence], axis=0).astype(np.float32)


def build_feature_matrix(
    series: dict[str, np.ndarray],
) -> np.ndarray:
    """Monta matriz (N, FEATURE_DIM) com uma linha de features por barra."""
    n = len(series["log_return"])
    rows = [build_feature_row(series, i) for i in range(n)]
    return np.stack(rows, axis=0).astype(np.float32)


def build_sequence_tensor(
    prices: np.ndarray,
    lookback: int,
    end_index: int,
    *,
    granularity: int = 60,
    symbol: str = "OTC_SPC",
    open_: np.ndarray | None = None,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    micro: dict[str, np.ndarray] | None = None,
    implied_vol_bars: int = 60,
) -> np.ndarray:
    """Monta janela (lookback, FEATURE_DIM) terminando em end_index inclusive."""
    series = precompute_price_series(
        prices,
        granularity=granularity,
        symbol=symbol,
        open_=open_,
        high=high,
        low=low,
        micro=micro,
        implied_vol_bars=implied_vol_bars,
    )
    start = end_index - lookback + 1
    rows = [build_feature_row(series, i) for i in range(start, end_index + 1)]
    return np.stack(rows, axis=0).astype(np.float32)
