"""Logger obrigatorio dos indicadores M5, decisao e probabilidades do Laya."""

from __future__ import annotations

from typing import Any

from src.domain.indicators.technical_indicators import IndicatorSnapshot
from src.domain.models.laya_decision import LayaDecision
from src.presentation.terminal.log_context import bind_log_context, get_log_context, restore_log_context


def log_laya_cycle_decision(
    orch: Any,
    snapshot: IndicatorSnapshot,
    decision: LayaDecision,
    stake: float,
    stake_pct: float,
    balance: float,
) -> None:
    """Emite no logger AETH todos os indicadores, direcao, probabilidades e stake."""
    prev_cid, prev_sym = get_log_context()
    bind_log_context(cycle_id=int(getattr(orch, "_active_cycle_id", 0) or 0), symbol=decision.symbol)
    try:
        orch.logger.info("")
        orch.logger.info(
            "INDICATORS | %s M5 close=%.2f | EMA20/50=%.2f/%.2f | RSI14=%.1f | MACD_h=%+.4f | "
            "ADX14=%.1f | Stoch=%.1f | BB%%B=%.2f | ATR14=%.2f | Hurst=%.2f | streak=%+d",
            decision.symbol,
            snapshot.close,
            snapshot.ema_20,
            snapshot.ema_50,
            snapshot.rsi_14,
            snapshot.macd_hist,
            snapshot.adx_14,
            snapshot.stoch_k,
            snapshot.bollinger_pct_b,
            snapshot.atr_14,
            snapshot.hurst_14,
            snapshot.streak,
        )
        orch.logger.info(
            "INDICATORS_FULL | %s M5 close=%.4f | EMA9=%.4f EMA20=%.4f EMA21=%.4f EMA50=%.4f "
            "spread=%.3fATR sl20=%.3f sl50=%.3f | RSI7=%.2f RSI14=%.2f | "
            "MACD=%.4f sig=%.4f hist=%.4f dHist=%+.4f | ADX14=%.2f (+DI=%.2f -DI=%.2f dDI=%+.2f) "
            "ADX20=%.2f dDI20=%+.2f | CCI14=%.1f CMO14=%.1f Stoch=%.1f/%.1f WR=%.1f ROC=%.3f | "
            "BB%%B=%.3f BBw=%.4f KC%%B=%.3f | ATR14=%.4f ATR50=%.4f ratio=%.3f | "
            "Hurst=%.3f Z=%.2f body=%.2fATR wick=%.2f/%.2f dir=%+d streak=%+d",
            decision.symbol,
            snapshot.close,
            snapshot.ema_9,
            snapshot.ema_20,
            snapshot.ema_21,
            snapshot.ema_50,
            snapshot.ema_spread_atr,
            snapshot.ema20_slope,
            snapshot.ema50_slope,
            snapshot.rsi_7,
            snapshot.rsi_14,
            snapshot.macd,
            snapshot.macd_signal,
            snapshot.macd_hist,
            snapshot.macd_hist_delta,
            snapshot.adx_14,
            snapshot.plus_di_14,
            snapshot.minus_di_14,
            snapshot.di_diff_14,
            snapshot.adx_20,
            snapshot.di_diff_20,
            snapshot.cci_14,
            snapshot.cmo_14,
            snapshot.stoch_k,
            snapshot.stoch_d,
            snapshot.williams_r_14,
            snapshot.roc_14,
            snapshot.bollinger_pct_b,
            snapshot.bollinger_width,
            snapshot.keltner_pct_b,
            snapshot.atr_14,
            snapshot.atr_50,
            snapshot.atr_ratio,
            snapshot.hurst_14,
            snapshot.session_zscore,
            snapshot.body_atr,
            snapshot.upper_wick_atr,
            snapshot.lower_wick_atr,
            snapshot.candle_dir,
            snapshot.streak,
        )
        orch.logger.info(
            "LAYA_DECISION | %s M5 dir=%s | P(CALL)=%.6f%% P(PUT)=%.6f%% | prob=%.6f conf=%.6f | "
            "lat=%.0fms | stake=$%.2f (%.2f%%) | banca=$%.2f",
            decision.symbol,
            decision.direction.name,
            decision.prob_call * 100.0,
            decision.prob_put * 100.0,
            decision.probability,
            decision.confidence,
            decision.latency_ms,
            stake,
            stake_pct * 100.0,
            balance,
        )
    finally:
        restore_log_context(prev_cid, prev_sym)
