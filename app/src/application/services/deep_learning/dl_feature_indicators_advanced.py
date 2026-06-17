"""Indicadores tecnicos avancados normalizados para features DL."""

import numpy as np
import polars as pl


def calculate_adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula o ADX e a diferenca normalizada (+DI - -DI) / 100.0."""
    n = len(close)
    adx_out = np.zeros(n, dtype=np.float64)
    di_diff_out = np.zeros(n, dtype=np.float64)
    if n < period + 1:
        return adx_out, di_diff_out

    df = pl.DataFrame(
        {
            "high": high.astype(np.float64),
            "low": low.astype(np.float64),
            "close": close.astype(np.float64),
        }
    )

    span = 2 * period - 1

    df = df.with_columns(
        [
            pl.col("close").shift(1).alias("prev_close"),
            pl.col("high").shift(1).alias("prev_high"),
            pl.col("low").shift(1).alias("prev_low"),
        ]
    )

    df = df.with_columns(
        [
            (pl.col("high") - pl.col("low")).alias("tr1"),
            (pl.col("high") - pl.col("prev_close")).abs().alias("tr2"),
            (pl.col("low") - pl.col("prev_close")).abs().alias("tr3"),
        ]
    )
    df = df.with_columns(pl.max_horizontal("tr1", "tr2", "tr3").alias("tr"))

    df = df.with_columns(
        [
            (pl.col("high") - pl.col("prev_high")).alias("up_move"),
            (pl.col("prev_low") - pl.col("low")).alias("down_move"),
        ]
    )

    df = df.with_columns(
        [
            pl.when((pl.col("up_move") > pl.col("down_move")) & (pl.col("up_move") > 0.0))
            .then(pl.col("up_move"))
            .otherwise(0.0)
            .alias("plus_dm"),
            pl.when((pl.col("down_move") > pl.col("up_move")) & (pl.col("down_move") > 0.0))
            .then(pl.col("down_move"))
            .otherwise(0.0)
            .alias("minus_dm"),
        ]
    )

    df = df.with_columns(
        [
            pl.col("tr").ewm_mean(span=span, adjust=False).alias("tr_smooth"),
            pl.col("plus_dm").ewm_mean(span=span, adjust=False).alias("plus_dm_smooth"),
            pl.col("minus_dm").ewm_mean(span=span, adjust=False).alias("minus_dm_smooth"),
        ]
    )

    df = df.with_columns(
        [
            (100.0 * pl.col("plus_dm_smooth") / (pl.col("tr_smooth") + 1e-10)).alias("plus_di"),
            (100.0 * pl.col("minus_dm_smooth") / (pl.col("tr_smooth") + 1e-10)).alias("minus_di"),
        ]
    )

    df = df.with_columns(
        (
            100.0 * (pl.col("plus_di") - pl.col("minus_di")).abs() / (pl.col("plus_di") + pl.col("minus_di") + 1e-10)
        ).alias("dx")
    )
    df = df.with_columns(pl.col("dx").ewm_mean(span=span, adjust=False).alias("adx"))

    adx_res = df.select("adx").to_numpy().flatten()
    plus_di_res = df.select("plus_di").to_numpy().flatten()
    minus_di_res = df.select("minus_di").to_numpy().flatten()

    adx_out = adx_res / 100.0
    di_diff_out = (plus_di_res - minus_di_res) / 100.0

    return np.nan_to_num(adx_out, nan=0.0, posinf=0.0, neginf=0.0), np.nan_to_num(
        di_diff_out, nan=0.0, posinf=0.0, neginf=0.0
    )


def calculate_williams_r(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Calcula Williams %R normalizado na escala de 0.0 a 1.0."""
    n = len(close)
    out = np.full(n, 0.5, dtype=np.float64)
    if n < period:
        return out
    for i in range(period - 1, n):
        start = i - period + 1
        h_val = np.max(high[start : i + 1])
        l_val = np.min(low[start : i + 1])
        denom = h_val - l_val
        if denom > 1e-10:
            out[i] = (close[i] - l_val) / denom
        else:
            out[i] = 0.5
    return out


def calculate_volatility_ratio(
    log_return: np.ndarray,
    short: int = 5,
    long: int = 20,
) -> np.ndarray:
    """Razao de volatilidade realizada de curto prazo vs longo prazo."""
    n = len(log_return)
    out = np.zeros(n, dtype=np.float64)
    min_len = max(short, long)
    if n < min_len:
        return out
    for i in range(min_len - 1, n):
        seg_short = log_return[i - short + 1 : i + 1]
        seg_long = log_return[i - long + 1 : i + 1]
        std_short = np.std(seg_short)
        std_long = np.std(seg_long)
        if std_long > 1e-10:
            out[i] = std_short / std_long
        else:
            out[i] = 1.0
    return out


def calculate_ema_crossover(
    prices: np.ndarray,
    fast: int = 9,
    slow: int = 21,
) -> np.ndarray:
    """Calcula a distancia normalizada entre duas EMAs: (EMA_fast - EMA_slow) / close."""
    df = pl.DataFrame({"close": prices})
    ema_fast = df.select(pl.col("close").ewm_mean(span=int(fast))).to_numpy().flatten()
    ema_slow = df.select(pl.col("close").ewm_mean(span=int(slow))).to_numpy().flatten()
    dist = (ema_fast - ema_slow) / (prices + 1e-10)
    return np.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0)
