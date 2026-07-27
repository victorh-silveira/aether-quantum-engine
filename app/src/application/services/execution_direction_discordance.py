"""Discordancia tecnica RSI/DI e consenso de votos contra a direcao TCN."""

from __future__ import annotations

from src.domain.models.trade import TradeDirection


def _macro_indicator_float(metrics: dict, key: str) -> float | None:
    """Le indicador float priorizando macro (mesmo bloco da votacao TCN)."""
    for block_name in ("macro_indicators", "indicators"):
        block = metrics.get(block_name)
        if not isinstance(block, dict) or block.get(key) is None:
            continue
        try:
            return float(block[key])
        except (TypeError, ValueError):
            return None
    return None


def _rsi_di_oppose_direction(metrics: dict, dl_dir: TradeDirection) -> bool:
    """True quando RSI e di_diff macro votam contra a direcao TCN com vies claro."""
    rsi = _macro_indicator_float(metrics, "rsi")
    di_diff = _macro_indicator_float(metrics, "di_diff")
    if rsi is None:
        return False
    rsi_bias = float(rsi) - 0.5
    if abs(rsi_bias) < 0.01:
        return False
    rsi_call = rsi_bias > 0.0
    want_call = dl_dir == TradeDirection.CALL
    if di_diff is not None:
        di_call = float(di_diff) > 0.0
        return want_call not in {rsi_call, di_call}
    return want_call != rsi_call


def resolve_formed_candle_direction(metrics: dict, default_dir: TradeDirection) -> TradeDirection:
    """Resolve a direcao alinhada com a cor/corpo da vela formada e existente."""
    open_p = _macro_indicator_float(metrics, "open") or _macro_indicator_float(metrics, "open_price")
    close_p = (
        _macro_indicator_float(metrics, "close")
        or _macro_indicator_float(metrics, "close_price")
        or _macro_indicator_float(metrics, "last_price")
    )
    if open_p is not None and close_p is not None:
        candle_dir = TradeDirection.CALL if close_p >= open_p else TradeDirection.PUT
        metrics["candle_color_direction"] = candle_dir.name
        return candle_dir
    return default_dir


def align_direction_to_rsi_trend(dl_dir: TradeDirection, metrics: dict) -> TradeDirection:
    """Alinha a direcao como um Trader Senior Institucional (RSI > 0.50 -> CALL, RSI < 0.50 -> PUT, filtro ADX/Vela)."""
    rsi = _macro_indicator_float(metrics, "rsi")
    candle_dir = resolve_formed_candle_direction(metrics, dl_dir)
    if rsi is None:
        return candle_dir
    rsi_v = float(rsi)
    adx = _macro_indicator_float(metrics, "adx")
    hurst = _macro_indicator_float(metrics, "hurst")
    adx_v = float(adx) if adx is not None else 0.20
    hurst_v = float(hurst) if hurst is not None else 0.40
    if adx_v < 0.20 and hurst_v < 0.45:
        metrics["micro_regime_mean_reversion"] = True

    if 0.48 <= rsi_v <= 0.52:
        metrics["senior_trader_regime"] = "chop_candle_alignment"
        return candle_dir

    rsi_dir = TradeDirection.CALL if rsi_v > 0.50 else TradeDirection.PUT

    if adx_v >= 0.22 and hurst_v >= 0.50:
        metrics["senior_trader_regime"] = "strong_trend_impulse"
    else:
        metrics["senior_trader_regime"] = "momentum_alignment"

    if dl_dir != rsi_dir:
        metrics["rsi_trend_flipped"] = True
        metrics["rsi_trend_orig"] = dl_dir.name
        metrics["rsi_val"] = float(rsi)
        return rsi_dir
    return dl_dir


def apply_technical_agreement(metrics: dict, dl_dir: TradeDirection, prob: float, exec_cfg: dict) -> tuple[float, bool]:
    """Ajusta probabilidade e aplica bonus/penalidade por confluencia tecnica de indicadores."""
    call_votes, put_votes = int(metrics.get("call_votes", 0)), int(metrics.get("put_votes", 0))
    total = call_votes + put_votes
    opp = 0.0
    adjusted = prob
    if total > 0:
        confluence = (call_votes / total) if dl_dir == TradeDirection.CALL else (put_votes / total)
        opp = 1.0 - confluence
        metrics["indicator_confluence_ratio"] = round(confluence, 3)
        if confluence >= 0.80:
            bonus = 0.06
            adjusted = min(1.0, prob + bonus) if dl_dir == TradeDirection.CALL else max(0.0, prob - bonus)
            metrics["indicator_confluence_bonus"] = bonus
        elif confluence >= 0.60:
            bonus = 0.03
            adjusted = min(1.0, prob + bonus) if dl_dir == TradeDirection.CALL else max(0.0, prob - bonus)
            metrics["indicator_confluence_bonus"] = bonus
        elif confluence < 0.35:
            penalty = 0.04
            adjusted = max(0.0, prob - penalty) if dl_dir == TradeDirection.CALL else min(1.0, prob + penalty)
            metrics["indicator_confluence_penalty"] = penalty

    discordance_enabled = bool(exec_cfg.get("discordance_veto_enabled", False))
    vote_veto = bool(discordance_enabled and total >= 3 and opp + 1e-12 >= 0.60)
    trend_name = str(metrics.get("trend_direction") or "").upper()
    consensus_required = bool(exec_cfg.get("require_indicator_consensus", False))
    dt = exec_cfg.get("dynamic_threshold")
    if isinstance(dt, dict) and dt.get("require_indicator_consensus") is not None:
        consensus_required = bool(dt.get("require_indicator_consensus"))
    trend_veto = bool(
        discordance_enabled and consensus_required and trend_name in {"CALL", "PUT"} and trend_name != dl_dir.name
    )
    side_veto = bool(discordance_enabled and _rsi_di_oppose_direction(metrics, dl_dir))
    should_veto = bool(vote_veto or trend_veto or side_veto)
    if should_veto:
        metrics["gate_reason"] = "indicator_discordance"
        metrics["indicator_discordance_opp"] = float(opp)
        if side_veto:
            metrics["indicator_side_discordance"] = True
        if trend_veto:
            metrics["indicator_trend_discordance"] = True
    return adjusted, should_veto
