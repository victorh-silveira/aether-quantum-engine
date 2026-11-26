"""Estrutura imutavel de snapshot dos indicadores tecnicos causais."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class IndicatorSnapshot:
    """Fotografia causal de todos os indicadores tecnicos na ultima vela fechada."""

    close: float
    ema_9: float
    ema_20: float
    ema_21: float
    ema_50: float
    ema_spread_atr: float
    ema20_slope: float
    ema50_slope: float
    rsi_7: float
    rsi_14: float
    macd: float
    macd_signal: float
    macd_hist: float
    macd_hist_delta: float
    stoch_k: float
    stoch_d: float
    cci_14: float
    cmo_14: float
    williams_r_14: float
    roc_14: float
    adx_14: float
    plus_di_14: float
    minus_di_14: float
    di_diff_14: float
    adx_20: float
    di_diff_20: float
    atr_14: float
    atr_50: float
    atr_ratio: float
    bollinger_pct_b: float
    bollinger_width: float
    keltner_pct_b: float
    hurst_14: float
    body_atr: float
    upper_wick_atr: float
    lower_wick_atr: float
    candle_dir: int
    streak: int
    session_zscore: float

    def to_dict(self) -> dict[str, float]:
        """Converte todos os indicadores para dicionario serializavel."""
        return {key: float(val) for key, val in asdict(self).items()}
