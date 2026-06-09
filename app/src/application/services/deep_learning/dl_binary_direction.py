"""Filtros de opcoes binarias para indices sinteticos Deriv (random walk e mean reversion)."""

from typing import Any

import numpy as np

from src.application.services.deep_learning.dl_features import precompute_price_series
from src.application.services.deep_learning.dl_pair_features import precompute_pair_series
from src.domain.models.trade import TradeDirection


def variance_ratio(returns: np.ndarray, short: int = 5) -> float:
    """Razao de variancia multi-periodo vs single-periodo; ~1 em passeio aleatorio."""
    n = len(returns)
    step = max(2, int(short))
    if n < step + 5:
        return 1.0
    tail = returns[-(step + 20) :]
    single_var = float(np.var(tail)) + 1e-12
    agg = []
    for i in range(step, len(tail)):
        agg.append(float(np.sum(tail[i - step + 1 : i + 1])))
    return float(np.var(agg)) / (step * single_var)


def build_binary_context(
    prices: np.ndarray,
    *,
    granularity: int = 300,
    pair_prices: np.ndarray | None = None,
    sym_is_bull: bool = False,
    open_=None,
    high=None,
    low=None,
) -> dict[str, float]:
    """Monta indicadores estatisticos da ultima barra para gating binario."""
    if len(prices) < 6:
        return {}
    series = precompute_price_series(
        prices,
        granularity=granularity,
        open_=open_,
        high=high,
        low=low,
    )
    idx = len(prices) - 1
    ctx: dict[str, float] = {
        "sma_z": float(series["sma_dist"][idx]),
        "rel_vol": float(series["rel_vol"][idx]),
        "rsi": float(series["rsi"][idx]),
        "body": float(series["body"][idx]),
        "upper_wick": float(series["upper_wick"][idx]),
        "close_loc": float(series["close_loc"][idx]),
        "variance_ratio": variance_ratio(series["returns"]),
        "z_spread": 0.0,
        "lower_wick": 0.0,
        "has_pair": 0.0,
    }
    if open_ is not None and low is not None and len(open_) > idx and len(low) > idx:
        o = float(open_[idx])
        low_px = float(low[idx])
        c = float(prices[idx])
        ctx["lower_wick"] = (min(o, c) - low_px) / (o + 1e-10)
    if pair_prices is not None and len(pair_prices) >= len(prices):
        bull, bear = (prices, pair_prices) if sym_is_bull else (pair_prices, prices)
        pair_series = precompute_pair_series(bull, bear)
        ctx["z_spread"] = float(pair_series["z_spread"][idx])
        ctx["has_pair"] = 1.0
    return ctx


def pair_spread_supports_direction(
    direction: TradeDirection,
    z_spread: float,
    *,
    sym_is_bull: bool,
    against_limit: float,
) -> bool:
    """Confirma CALL/PUT com desvio do spread log bull/bear no par Range."""
    limit = float(against_limit)
    if direction == TradeDirection.CALL:
        return z_spread >= -limit if sym_is_bull else z_spread <= limit
    return z_spread <= limit if sym_is_bull else z_spread >= -limit


def apply_mean_reversion_override(
    direction: TradeDirection,
    raw_prob: float,
    context: dict[str, float],
    params: dict[str, Any],
) -> tuple[TradeDirection, bool, float]:
    """Substitui direcao ambigua do DL por reversao a media em extremos de z-score."""
    bs = params.get("binary_signal") or {}
    extreme = float(bs.get("sma_z_extreme", 0.005))
    weak_margin = float(bs.get("weak_dl_override_margin", 0.05))
    sma_z = float(context.get("sma_z", 0.0))
    raw_side = max(float(raw_prob), 1.0 - float(raw_prob))
    if raw_side > 0.5 + weak_margin:
        return direction, False, raw_prob
    if sma_z >= extreme and direction != TradeDirection.PUT:
        return TradeDirection.PUT, True, 1.0 - float(raw_prob)
    if sma_z <= -extreme and direction != TradeDirection.CALL:
        return TradeDirection.CALL, True, 1.0 - float(raw_prob)
    return direction, False, raw_prob


def binary_direction_veto(
    direction: TradeDirection,
    context: dict[str, float],
    params: dict[str, Any],
    *,
    sym_is_bull: bool,
) -> str | None:
    """Retorna motivo de bloqueio ou None quando a direcao passa nos filtros binarios."""
    if not context:
        return None
    bs = params.get("binary_signal") or {}
    rel_vol = float(context.get("rel_vol", 0.0))
    sma_z = float(context.get("sma_z", 0.0))
    vr = float(context.get("variance_ratio", 1.0))
    body = float(context.get("body", 0.0))
    upper = float(context.get("upper_wick", 0.0))
    lower = float(context.get("lower_wick", 0.0))
    min_rel_vol = float(bs.get("min_rel_vol_execute", 0.28))
    block_call = float(bs.get("sma_z_block_call", 0.003))
    block_put = float(bs.get("sma_z_block_put", -0.003))
    vr_max = float(bs.get("variance_ratio_mean_rev_max", 0.88))
    wick_ratio = float(bs.get("wick_rejection_ratio", 1.8))
    body_abs = abs(body) + 1e-10
    reason = None
    if rel_vol + 1e-12 < min_rel_vol:
        reason = "noise_floor"
    elif (direction == TradeDirection.CALL and sma_z >= block_call) or (
        direction == TradeDirection.PUT and sma_z <= block_put
    ):
        reason = "mean_reversion"
    elif vr <= vr_max and (
        (direction == TradeDirection.CALL and sma_z > 0.0) or (direction == TradeDirection.PUT and sma_z < 0.0)
    ):
        reason = "random_walk"
    elif (
        bool(bs.get("require_pair_spread_confirm", True))
        and context.get("has_pair", 0.0) > 0.5
        and not pair_spread_supports_direction(
            direction,
            float(context.get("z_spread", 0.0)),
            sym_is_bull=sym_is_bull,
            against_limit=float(bs.get("pair_z_against_limit", 1.0)),
        )
    ):
        reason = "pair_spread"
    elif (direction == TradeDirection.CALL and body > 0.0 and upper > wick_ratio * body_abs) or (
        direction == TradeDirection.PUT and body < 0.0 and lower > wick_ratio * body_abs
    ):
        reason = "wick_reject"
    elif bool(bs.get("require_candle_confirm", True)):
        close_loc = float(context.get("close_loc", 0.5))
        min_call_loc = float(bs.get("min_close_loc_call", 0.48))
        max_put_loc = float(bs.get("max_close_loc_put", 0.52))
        if (
            direction == TradeDirection.CALL
            and (body <= 0.0 or close_loc < min_call_loc)
            or direction == TradeDirection.PUT
            and (body >= 0.0 or close_loc > max_put_loc)
        ):
            reason = "candle_reject"
    if reason is None:
        rsi = float(context.get("rsi", 0.5))
        rsi_block_call = float(bs.get("rsi_block_call", 0.72))
        rsi_block_put = float(bs.get("rsi_block_put", 0.28))
        if (
            direction == TradeDirection.CALL
            and rsi >= rsi_block_call
            or direction == TradeDirection.PUT
            and rsi <= rsi_block_put
        ):
            reason = "rsi_exhaust"
    return reason
