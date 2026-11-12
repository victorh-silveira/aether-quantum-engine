"""Filtros e formatadores de metricas para selecao de ordens do indicador."""

from __future__ import annotations

import math
import time
from typing import Any

from src.application.services.indicator_consensus import signal_epoch_current
from src.domain.symbols.drift_symbols import is_otc_exchange_open_brt, session_liquidity_weight


_DIRECT_REASONS = (
    "TREND_ALIGNED_DIRECT",
    "TELEMETRY_WIN_DIRECT",
    "LATERAL_FADE_INVERTED",
    "ELASTIC_EXHAUSTION_INVERTED",
)


def order_selection_score(orch: Any, symbol: str, metrics: dict, tier_adj: float) -> tuple:
    """Compoe tupla deterministica de ranqueamento entre ordens cotadas no ciclo."""
    l95, ev_val = metrics.get("indicator_broker_lower_95"), metrics.get("indicator_prequote_ev")
    rt = float(metrics.get("indicator_quote_rate") or 0.0)
    t_str = float(metrics.get("trend_strength") or 0.0)
    pen = -0.06 if getattr(orch, "_last_traded_symbol", None) == symbol else 0.0
    reason = str(metrics.get("organic_decision_reason") or "")
    bonus_direct = 0.08 if reason in _DIRECT_REASONS else 0.0
    trend_adj = 0.05 if t_str >= 25.0 else (-0.05 if t_str < 18.0 else 0.0)
    sess_trk = getattr(orch, "_session_pair_tracker", None)
    sess_adj = (
        sess_trk.symbol_rotation_adjustment(symbol)
        if sess_trk is not None and hasattr(sess_trk, "symbol_rotation_adjustment")
        else 0.0
    )
    liq_w = session_liquidity_weight(symbol, time.time())
    prob_term = float(metrics.get("trade_score") or metrics.get("calibrated_prob") or 0.5)
    return (
        float(l95) if l95 is not None else -math.inf,
        prob_term + pen + bonus_direct + trend_adj + tier_adj + sess_adj + (0.05 * liq_w),
        float(ev_val) if ev_val is not None else rt,
        t_str,
        rt,
    )


def closed_candles_slice(candles: list[Any], period: int, now_ts: float) -> list[Any]:
    """Descarta a ultima vela caso pertencente ao bloco atual ainda em formacao."""
    ep = int(getattr(candles[-1], "epoch", 0) or 0) if candles else 0
    return (
        candles[:-1]
        if len(candles) > 50 and period > 0 and now_ts > 100000 and ep >= int(now_ts // period) * period
        else candles
    )


def build_indicator_metrics(
    epoch: int,
    period: int,
    version: str,
    side: str,
    sig_side: str,
    reason: str,
    *,
    pilot: bool,
    ev: Any,
    prob: float,
) -> dict[str, Any]:
    """Cria payload padronizado de metricas para o contrato."""
    return {
        "indicator_controls_execution": True,
        "indicator_candle_epoch": epoch,
        "indicator_period_seconds": period,
        "allowed_period_seconds": period,
        "indicator_model_version": version,
        "indicator_call_probability": None,
        "indicator_execution_side": side,
        "indicator_signal_side": sig_side,
        "organic_decision_reason": reason,
        "indicator_demo_pilot": pilot,
        "indicator_broker_settled": getattr(ev, "settled", 0),
        "indicator_broker_lower_95": getattr(ev, "lower_95", None),
        "indicator_exploratory_purchased": getattr(ev, "exploratory_purchased", 0),
        "calibrated_prob": prob if side == "CALL" else 1.0 - prob,
        "raw_prob": None,
        "trade_score": prob,
        "conviction": prob,
        "senior_trader_conviction": prob,
        "direction_origin": "PULLBACK_SETUP",
        "indicator_tcn_discord": True,
        "duration": period // 60,
    }


def is_promoted_signal_viable(
    orch: Any,
    promoted_only: str | None,
    period_seconds: int,
    now_ts: float,
    signal_epoch_fn: Any = signal_epoch_current,
) -> bool:
    """Valida se o ativo promovido tem sinal valido e mercado aberto em Brasilia."""
    pb_cfg = orch.config.get("pullback_strategy", {})
    if not promoted_only or not is_otc_exchange_open_brt(str(promoted_only), now_ts):
        return False
    p_sig = getattr(orch, "_indicator_period_signals", {}).get((str(promoted_only), period_seconds))
    if not p_sig or p_sig.model_version != pb_cfg.get("version") or p_sig.side not in ("CALL", "PUT"):
        return False
    return bool(
        signal_epoch_fn(
            orch.config,
            {
                "indicator_controls_execution": True,
                "indicator_candle_epoch": p_sig.candle_epoch,
                "indicator_period_seconds": period_seconds,
            },
        )
    )
