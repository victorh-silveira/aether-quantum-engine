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
    if n < period + 1:
        return np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.float64)
    df = pl.DataFrame(
        {"high": high.astype(np.float64), "low": low.astype(np.float64), "close": close.astype(np.float64)}
    )
    span = 2 * period - 1
    df = (
        df.with_columns(
            prev_close=pl.col("close").shift(1),
            prev_high=pl.col("high").shift(1),
            prev_low=pl.col("low").shift(1),
        )
        .with_columns(
            tr1=pl.col("high") - pl.col("low"),
            tr2=(pl.col("high") - pl.col("prev_close")).abs(),
            tr3=(pl.col("low") - pl.col("prev_close")).abs(),
            up_move=pl.col("high") - pl.col("prev_high"),
            down_move=pl.col("prev_low") - pl.col("low"),
        )
        .with_columns(
            tr=pl.max_horizontal("tr1", "tr2", "tr3"),
            plus_dm=pl.when((pl.col("up_move") > pl.col("down_move")) & (pl.col("up_move") > 0.0))
            .then(pl.col("up_move"))
            .otherwise(0.0),
            minus_dm=pl.when((pl.col("down_move") > pl.col("up_move")) & (pl.col("down_move") > 0.0))
            .then(pl.col("down_move"))
            .otherwise(0.0),
        )
        .with_columns(
            tr_smooth=pl.col("tr").ewm_mean(span=span, adjust=False),
            plus_dm_smooth=pl.col("plus_dm").ewm_mean(span=span, adjust=False),
            minus_dm_smooth=pl.col("minus_dm").ewm_mean(span=span, adjust=False),
        )
        .with_columns(
            plus_di=100.0 * pl.col("plus_dm_smooth") / (pl.col("tr_smooth") + 1e-10),
            minus_di=100.0 * pl.col("minus_dm_smooth") / (pl.col("tr_smooth") + 1e-10),
        )
        .with_columns(
            dx=100.0 * (pl.col("plus_di") - pl.col("minus_di")).abs() / (pl.col("plus_di") + pl.col("minus_di") + 1e-10)
        )
        .with_columns(adx=pl.col("dx").ewm_mean(span=span, adjust=False))
    )
    adx_out = df.select("adx").to_numpy().flatten() / 100.0
    di_diff_out = (df.select("plus_di").to_numpy().flatten() - df.select("minus_di").to_numpy().flatten()) / 100.0
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


def calculate_cmo(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Calcula Chande Momentum Oscillator (CMO) normalizado na escala -1.0 a 1.0."""
    n = len(prices)
    out = np.zeros(n, dtype=np.float64)
    if n < period + 1:
        return out
    deltas = np.diff(prices)
    for i in range(period, n):
        segment = deltas[i - period : i]
        gains = float(segment[segment > 0].sum())
        losses = float(-segment[segment < 0].sum())
        denom = gains + losses
        if denom > 1e-10:
            out[i] = (gains - losses) / denom
        else:
            out[i] = 0.0
    return out


def calculate_keltner_channel_pct_b(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
    atr_period: int,
    *,
    atr_mult: float,
) -> np.ndarray:
    """Calcula Keltner Channel %b: (close - lower) / (upper - lower)."""
    n = len(close)
    out = np.full(n, 0.5, dtype=np.float64)
    if n < max(period, atr_period):
        return out
    df = pl.DataFrame({"close": close})
    ema = df.select(pl.col("close").ewm_mean(span=int(period))).to_numpy().flatten()

    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    df_tr = pl.DataFrame({"tr": tr})
    atr = df_tr.select(pl.col("tr").ewm_mean(span=int(atr_period))).to_numpy().flatten()
    mult = float(atr_mult)

    upper = ema + mult * atr
    lower = ema - mult * atr

    for i in range(n):
        denom = upper[i] - lower[i]
        if denom > 1e-10:
            out[i] = (close[i] - lower[i]) / denom
        else:
            out[i] = 0.5
    return out


def calculate_choppiness_index(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Calcula Choppiness Index (CI) normalizado em [0, 100]. High (>61.8) = consolidacao."""
    n = len(close)
    out = np.full(n, 50.0, dtype=np.float64)
    if n < period + 1:
        return out
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    p = max(2, int(period))
    log10_p = np.log10(p)
    for i in range(p, n):
        tr_sum = np.sum(tr[i - p + 1 : i + 1])
        max_h = np.max(high[i - p + 1 : i + 1])
        min_l = np.min(low[i - p + 1 : i + 1])
        rng = max_h - min_l
        if rng > 1e-10 and tr_sum > 0.0:
            ci = 100.0 * (np.log10(tr_sum / rng) / log10_p)
            out[i] = max(0.0, min(100.0, ci))
    return out


def calculate_vwap_zscore(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray | None = None,
    window: int = 20,
) -> np.ndarray:
    """Calcula Z-Score do desvio do preco em relacao ao VWAP rolling."""
    n = len(close)
    out = np.zeros(n, dtype=np.float64)
    if n < window:
        return out
    vol = volume if volume is not None and len(volume) == n else np.ones(n, dtype=np.float64)
    typical_price = (high + low + close) / 3.0
    w = max(2, int(window))
    for i in range(w - 1, n):
        tp_win = typical_price[i - w + 1 : i + 1]
        v_win = vol[i - w + 1 : i + 1]
        v_sum = np.sum(v_win)
        vwap = np.sum(tp_win * v_win) / (v_sum + 1e-10) if v_sum > 0 else np.mean(tp_win)
        std = np.std(tp_win)
        out[i] = (close[i] - vwap) / (std + 1e-10)
    return np.clip(out, -3.0, 3.0)


def calculate_supertrend(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 10,
    multiplier: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula linha de SuperTrend e sinal direcional (+1.0 CALL / -1.0 PUT)."""
    n = len(close)
    st = np.zeros(n, dtype=np.float64)
    dir_out = np.ones(n, dtype=np.float64)
    if n < period + 1:
        return st, dir_out
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    p = max(2, int(period))
    atr = np.zeros(n, dtype=np.float64)
    atr[p - 1] = np.mean(tr[:p])
    for i in range(p, n):
        atr[i] = (atr[i - 1] * (p - 1) + tr[i]) / p
    hl2 = (high + low) / 2.0
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    st_dir = 1
    for i in range(1, n):
        if close[i] > upper[i - 1]:
            st_dir = 1
        elif close[i] < lower[i - 1]:
            st_dir = -1
        if st_dir == 1:
            lower[i] = max(lower[i], lower[i - 1]) if close[i - 1] > lower[i - 1] else lower[i]
            st[i] = lower[i]
        else:
            upper[i] = min(upper[i], upper[i - 1]) if close[i - 1] < upper[i - 1] else upper[i]
            st[i] = upper[i]
        dir_out[i] = float(st_dir)
    return st, dir_out
