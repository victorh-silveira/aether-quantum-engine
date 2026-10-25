"""Filtros e formatadores de metricas para selecao de ordens do indicador."""

from __future__ import annotations

from typing import Any

from src.application.services.indicator_consensus import signal_epoch_current
from src.domain.symbols.drift_symbols import is_otc_exchange_open_brt


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
    if not promoted_only or not is_otc_exchange_open_brt(str(promoted_only), now_ts):
        return False
    p_sig = getattr(orch, "_indicator_period_signals", {}).get((str(promoted_only), period_seconds))
    if not p_sig or not p_sig.model_version.startswith("pullback-1-") or p_sig.side not in ("CALL", "PUT"):
        return False
    return bool(
        signal_epoch_fn(
            orch.config,
            {"indicator_candle_epoch": p_sig.candle_epoch, "indicator_period_seconds": period_seconds},
        )
    )
