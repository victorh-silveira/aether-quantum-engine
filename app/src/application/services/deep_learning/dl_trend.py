"""Cálculo consensual de direção de tendência no bridge Deep Learning."""

import numpy as np

from src.domain.models.trade import TradeDirection


def consensus_trend_direction(price_dir: TradeDirection, series: dict) -> tuple[TradeDirection, int, int]:
    """Votacao por consenso dos indicadores e features."""
    call_votes = 1 if price_dir == TradeDirection.CALL else 0
    put_votes = 1 if price_dir != TradeDirection.CALL else 0

    indicators = [
        ("di_diff", lambda x: float(x) > 0.0),
        (
            "macd",
            lambda x: (
                float(x) > float(series["macd_signal"][-1])
                if "macd_signal" in series and len(series["macd_signal"]) > 0
                else False
            ),
        ),
        ("rsi", lambda x: float(x) > 0.5),
        ("cmo", lambda x: float(x) > 0.0),
        ("keltner_pct_b", lambda x: float(x) > 0.5),
    ]

    for key, check in indicators:
        val = series.get(key)
        if val is not None and len(val) > 0:
            if check(val[-1]):
                call_votes += 1
            else:
                put_votes += 1

    return TradeDirection.CALL if call_votes >= put_votes else TradeDirection.PUT, call_votes, put_votes


def calculate_trend_direction(prices, series: dict, exec_cfg: dict) -> tuple[TradeDirection, str, int, int, int]:
    """Calcula a direcao da tendencia usando um consenso de multiplos indicadores tecnicos."""
    trend_period = int(exec_cfg.get("trend_period", 5))
    trend_use_ema = bool(exec_cfg.get("trend_use_ema", True))
    trend_use_slope = bool(exec_cfg.get("trend_use_slope", True))
    close_prices = prices.astype(np.float64)
    t_len = min(trend_period, len(close_prices))
    if t_len > 0:
        if trend_use_ema and t_len > 1:
            alpha = 2.0 / (t_len + 1)
            ema = close_prices[-t_len]
            for price in close_prices[-t_len + 1 :]:
                ema = alpha * price + (1.0 - alpha) * ema
            trend_val = ema
        else:
            trend_val = np.mean(close_prices[-t_len:])
    else:
        trend_val = close_prices[-1] if len(close_prices) > 0 else 0.0

    if trend_use_slope and len(close_prices) > 1:
        prev_prices = close_prices[:-1]
        prev_len = min(trend_period, len(prev_prices))
        if trend_use_ema and prev_len > 1:
            alpha = 2.0 / (prev_len + 1)
            prev_ema = prev_prices[-prev_len]
            for price in prev_prices[-prev_len + 1 :]:
                prev_ema = alpha * price + (1.0 - alpha) * prev_ema
            prev_trend_val = prev_ema
        else:
            prev_trend_val = np.mean(prev_prices[-prev_len:])
        price_dir = TradeDirection.CALL if trend_val >= prev_trend_val else TradeDirection.PUT
    else:
        last_val = close_prices[-1] if len(close_prices) > 0 else 0.0
        price_dir = TradeDirection.CALL if last_val >= trend_val else TradeDirection.PUT

    if len(close_prices) >= 30:
        trend_dir, call_votes, put_votes = consensus_trend_direction(price_dir, series)
        trend_type = "CONSENSUS"
    else:
        trend_dir = price_dir
        trend_type = "EMA" if trend_use_ema else "SMA"
        call_votes = 1 if price_dir == TradeDirection.CALL else 0
        put_votes = 1 if price_dir != TradeDirection.CALL else 0
    return trend_dir, trend_type, trend_period, call_votes, put_votes
