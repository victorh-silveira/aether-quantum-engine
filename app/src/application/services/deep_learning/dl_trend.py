"""Calculo consensual de direcao de tendencia no bridge Deep Learning."""

import json

import numpy as np

from aether_paths import repo_path
from src.application.services.deep_learning.dl_indicator_config import load_indicator_config_from_settings
from src.domain.models.trade import TradeDirection


def _safe_last(series: dict, key: str) -> float | None:
    """Extrai o ultimo valor de uma serie de indicadores de forma thread-safe e sem race conditions."""
    val = series.get(key)
    if val is not None and len(val) > 0:
        return float(val[-1])
    return None


def consensus_trend_direction(
    price_dir: TradeDirection,
    series: dict,
    trend_consensus: dict,
) -> tuple[TradeDirection, int, int]:
    """Votacao por consenso dos indicadores tecnicos com zonas puras CALL, PUT e NEUTRA sem race condition."""
    call_votes = 1 if price_dir == TradeDirection.CALL else 0
    put_votes = 1 if price_dir != TradeDirection.CALL else 0
    rsi_above = float(trend_consensus.get("rsi_call_above", 0.50))
    keltner_above = float(trend_consensus.get("keltner_call_above", 0.50))
    di_above = float(trend_consensus.get("di_call_above", 0.0))

    # 1. DI Diff (Zona Neutra: [-0.02, +0.02])
    v = _safe_last(series, "di_diff")
    if v is not None:
        if v > (di_above + 0.02):
            call_votes += 1
        elif v < (di_above - 0.02):
            put_votes += 1

    # 2. MACD Diff (Zona Neutra: [-0.00005, +0.00005])
    m_val = _safe_last(series, "macd")
    m_sig = _safe_last(series, "macd_signal")
    if m_val is not None and m_sig is not None:
        diff = m_val - m_sig
        if diff > 0.00005:
            call_votes += 1
        elif diff < -0.00005:
            put_votes += 1

    # 3. RSI (Zona Neutra: [0.48, 0.52])
    rv = _safe_last(series, "rsi")
    if rv is not None:
        if rv > max(rsi_above, 0.52):
            call_votes += 1
        elif rv < min(rsi_above, 0.48):
            put_votes += 1

    # 4. CMO (Zona Neutra: [-0.05, +0.05])
    cv = _safe_last(series, "cmo")
    if cv is not None:
        if cv > 0.05:
            call_votes += 1
        elif cv < -0.05:
            put_votes += 1

    # 5. Keltner %B (Zona Neutra: [0.48, 0.52])
    kv = _safe_last(series, "keltner_pct_b")
    if kv is not None:
        if kv > (keltner_above + 0.02):
            call_votes += 1
        elif kv < (keltner_above - 0.02):
            put_votes += 1

    return TradeDirection.CALL if call_votes >= put_votes else TradeDirection.PUT, call_votes, put_votes


def _ema_tail(prices: np.ndarray, period: int) -> float:
    """EMA ou media da cauda de tamanho period."""
    t_len = min(period, len(prices))
    if t_len <= 0:
        return float(prices[-1]) if len(prices) > 0 else 0.0
    if t_len == 1:
        return float(prices[-1])
    alpha = 2.0 / (t_len + 1)
    ema = prices[-t_len]
    for price in prices[-t_len + 1 :]:
        ema = alpha * price + (1.0 - alpha) * ema
    return float(ema)


def _sma_tail(prices: np.ndarray, period: int) -> float:
    """Media simples da cauda."""
    t_len = min(period, len(prices))
    if t_len <= 0:
        return float(prices[-1]) if len(prices) > 0 else 0.0
    return float(np.mean(prices[-t_len:]))


def _load_execution_trend_defaults() -> dict:
    """Le trend_* de settings.orchestrator.execution."""
    path = repo_path("config", "settings.json")
    with path.open(encoding="utf-8") as handle:
        full = json.load(handle)
    return (full.get("orchestrator") or {}).get("execution") or {}


def calculate_trend_direction(prices, series: dict, exec_cfg: dict) -> tuple[TradeDirection, str, int, int, int]:
    """Calcula a direcao da tendencia usando um consenso de multiplos indicadores tecnicos."""
    defaults = _load_execution_trend_defaults()
    if "trend_period" not in exec_cfg and "trend_period" not in defaults:
        raise KeyError("orchestrator.execution.trend_period obrigatorio")
    trend_period = int(exec_cfg["trend_period"] if "trend_period" in exec_cfg else defaults["trend_period"])
    trend_use_ema = bool(
        exec_cfg["trend_use_ema"] if "trend_use_ema" in exec_cfg else defaults.get("trend_use_ema", True)
    )
    trend_use_slope = bool(
        exec_cfg["trend_use_slope"] if "trend_use_slope" in exec_cfg else defaults.get("trend_use_slope", True)
    )
    trend_consensus = load_indicator_config_from_settings()["trend_consensus"]
    min_bars = int(trend_consensus["min_bars"])
    close_prices = prices.astype(np.float64)
    trend_val = _ema_tail(close_prices, trend_period) if trend_use_ema else _sma_tail(close_prices, trend_period)
    if trend_use_slope and len(close_prices) > 1:
        prev = close_prices[:-1]
        prev_val = _ema_tail(prev, trend_period) if trend_use_ema else _sma_tail(prev, trend_period)
        price_dir = TradeDirection.CALL if trend_val >= prev_val else TradeDirection.PUT
    else:
        last_val = close_prices[-1] if len(close_prices) > 0 else 0.0
        price_dir = TradeDirection.CALL if last_val >= trend_val else TradeDirection.PUT
    if len(close_prices) >= min_bars:
        trend_dir, call_votes, put_votes = consensus_trend_direction(price_dir, series, trend_consensus)
        return trend_dir, "CONSENSUS", trend_period, call_votes, put_votes
    call_votes = 1 if price_dir == TradeDirection.CALL else 0
    put_votes = 1 if price_dir != TradeDirection.CALL else 0
    return price_dir, ("EMA" if trend_use_ema else "SMA"), trend_period, call_votes, put_votes
