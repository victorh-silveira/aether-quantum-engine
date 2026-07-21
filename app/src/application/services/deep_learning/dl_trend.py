"""Calculo consensual de direcao de tendencia no bridge Deep Learning."""

import json

import numpy as np

from aether_paths import repo_path
from src.application.services.deep_learning.dl_indicator_config import load_indicator_config_from_settings
from src.domain.models.trade import TradeDirection


def consensus_trend_direction(
    price_dir: TradeDirection,
    series: dict,
    trend_consensus: dict,
) -> tuple[TradeDirection, int, int]:
    """Votacao por consenso dos indicadores e features."""
    call_votes = 1 if price_dir == TradeDirection.CALL else 0
    put_votes = 1 if price_dir != TradeDirection.CALL else 0
    rsi_above = float(trend_consensus["rsi_call_above"])
    keltner_above = float(trend_consensus["keltner_call_above"])
    di_above = float(trend_consensus["di_call_above"])

    indicators = [
        ("di_diff", lambda x: float(x) > di_above),
        (
            "macd",
            lambda x: (
                float(x) > float(series["macd_signal"][-1])
                if "macd_signal" in series and len(series["macd_signal"]) > 0
                else False
            ),
        ),
        ("rsi", lambda x: float(x) > rsi_above),
        ("cmo", lambda x: float(x) > 0.0),
        ("keltner_pct_b", lambda x: float(x) > keltner_above),
    ]

    for key, check in indicators:
        val = series.get(key)
        if val is not None and len(val) > 0:
            if check(val[-1]):
                call_votes += 1
            else:
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
