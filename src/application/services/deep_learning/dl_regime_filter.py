"""Filtro de regime de mercado e deteccao de desequilibrio causal para TCN."""

from __future__ import annotations

import numpy as np


def market_disequilibrium_polarity(
    series: dict[str, np.ndarray],
    index: int,
    *,
    bb_lower_thr: float = 0.04,
    bb_upper_thr: float = 0.96,
    kc_lower_thr: float = 0.03,
    kc_upper_thr: float = 0.97,
    rsi_oversold: float = 0.20,
    rsi_overbought: float = 0.80,
    rsi_delta_thr: float = 0.04,
    ema_stretch_thr: float = 1.6,
    stoch_lower_thr: float = 0.15,
    stoch_upper_thr: float = 0.85,
    macd_atr_mult: float = 1.1,
    min_confluence_score: int = 2,
) -> int:
    """Retorna polaridade causal: +1 (sobrevenda/CALL), -1 (sobrecompra/PUT) ou 0 (neutro)."""
    if not series:
        return 0

    score_oversold = 0
    score_overbought = 0

    has_bb = "bb_pct_b" in series
    has_kc = "keltner_pct_b" in series
    if has_bb and index < len(series["bb_pct_b"]):
        val_bb = float(series["bb_pct_b"][index])
        if val_bb <= bb_lower_thr:
            score_oversold += 1
        elif val_bb >= bb_upper_thr:
            score_overbought += 1
    elif has_kc and index < len(series["keltner_pct_b"]):
        val_kc = float(series["keltner_pct_b"][index])
        if val_kc <= kc_lower_thr:
            score_oversold += 1
        elif val_kc >= kc_upper_thr:
            score_overbought += 1

    if "rsi" in series and index < len(series["rsi"]):
        val_rsi = float(series["rsi"][index])
        if val_rsi <= rsi_oversold:
            score_oversold += 1
        elif val_rsi >= rsi_overbought:
            score_overbought += 1

    if "delta_rsi" in series and index < len(series["delta_rsi"]):
        val_delta = float(series["delta_rsi"][index])
        if val_delta <= -rsi_delta_thr:
            score_oversold += 1
        elif val_delta >= rsi_delta_thr:
            score_overbought += 1

    if "ema_9_21_dist" in series and index < len(series["ema_9_21_dist"]):
        atr_val = float(series["atr_raw"][index]) if "atr_raw" in series and index < len(series["atr_raw"]) else 0.001
        dist = float(series["ema_9_21_dist"][index])
        stretch = abs(dist) / max(1e-6, atr_val)
        if stretch >= ema_stretch_thr:
            if dist < 0.0:
                score_oversold += 1
            else:
                score_overbought += 1

    if "stoch_k" in series and index < len(series["stoch_k"]):
        val_stoch = float(series["stoch_k"][index])
        if val_stoch <= stoch_lower_thr:
            score_oversold += 1
        elif val_stoch >= stoch_upper_thr:
            score_overbought += 1

    if "macd" in series and "macd_signal" in series and index < len(series["macd"]):
        hist = float(series["macd"][index]) - float(series["macd_signal"][index])
        atr_val = float(series["atr_raw"][index]) if "atr_raw" in series and index < len(series["atr_raw"]) else 0.001
        if abs(hist) >= macd_atr_mult * max(1e-6, atr_val):
            if hist < 0.0:
                score_oversold += 1
            else:
                score_overbought += 1

    target_score = max(1, int(min_confluence_score))
    if score_oversold >= target_score and score_oversold > score_overbought:
        return 1
    if score_overbought >= target_score and score_overbought > score_oversold:
        return -1
    return 0


def is_market_disequilibrium(
    series: dict[str, np.ndarray],
    index: int,
    *,
    bb_lower_thr: float = 0.04,
    bb_upper_thr: float = 0.96,
    kc_lower_thr: float = 0.03,
    kc_upper_thr: float = 0.97,
    rsi_oversold: float = 0.20,
    rsi_overbought: float = 0.80,
    rsi_delta_thr: float = 0.04,
    ema_stretch_thr: float = 1.6,
    stoch_lower_thr: float = 0.15,
    stoch_upper_thr: float = 0.85,
    macd_atr_mult: float = 1.1,
    min_confluence_score: int = 2,
) -> bool:
    """Verifica se a barra index possui desequilibrio direcional causal no sweet spot final."""
    if not series:
        return True
    return (
        market_disequilibrium_polarity(
            series,
            index,
            bb_lower_thr=bb_lower_thr,
            bb_upper_thr=bb_upper_thr,
            kc_lower_thr=kc_lower_thr,
            kc_upper_thr=kc_upper_thr,
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought,
            rsi_delta_thr=rsi_delta_thr,
            ema_stretch_thr=ema_stretch_thr,
            stoch_lower_thr=stoch_lower_thr,
            stoch_upper_thr=stoch_upper_thr,
            macd_atr_mult=macd_atr_mult,
            min_confluence_score=min_confluence_score,
        )
        != 0
    )


def is_price_disequilibrium(
    prices: np.ndarray,
    index: int,
    *,
    window: int = 60,
    z_threshold: float = 2.5,
) -> bool:
    """Deteccao de desequilibrio baseada unicamente na serie temporal de precos (>2.5 sigma na janela de 60 barras)."""
    if len(prices) < 2 or index < 1:
        return True
    start = max(0, index - max(2, int(window)))
    seg = prices[start : index + 1]
    if len(seg) < 3:
        return True
    mean_val = float(np.mean(seg))
    std_val = float(np.std(seg))
    if std_val < 1e-9:
        return False
    z_score = abs(float(prices[index]) - mean_val) / std_val
    return z_score >= z_threshold
