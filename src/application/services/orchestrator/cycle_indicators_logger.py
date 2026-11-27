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
        orch.logger.info("------------------------------------------------------------------------")
        orch.logger.info("INDICATORS | %s M5 | close=%.2f", decision.symbol, snapshot.close)
        orch.logger.info(
            "  TENDENCIA    | EMA9=%.2f | EMA20=%.2f | EMA21=%.2f | EMA50=%.2f | "
            "spread=%.2fATR | sl20=%.2f | sl50=%.2f",
            snapshot.ema_9,
            snapshot.ema_20,
            snapshot.ema_21,
            snapshot.ema_50,
            snapshot.ema_spread_atr,
            snapshot.ema20_slope,
            snapshot.ema50_slope,
        )
        orch.logger.info(
            "  MOMENTUM     | RSI7=%.2f | RSI14=%.2f | MACD=%.2f | sig=%.2f | hist=%+.2f | dHist=%+.2f",
            snapshot.rsi_7,
            snapshot.rsi_14,
            snapshot.macd,
            snapshot.macd_signal,
            snapshot.macd_hist,
            snapshot.macd_hist_delta,
        )
        orch.logger.info(
            "  DIRECAO      | ADX14=%.2f (+DI=%.2f -DI=%.2f dDI=%+.2f) | ADX20=%.2f (dDI20=%+.2f)",
            snapshot.adx_14,
            snapshot.plus_di_14,
            snapshot.minus_di_14,
            snapshot.di_diff_14,
            snapshot.adx_20,
            snapshot.di_diff_20,
        )
        orch.logger.info(
            "  OSCILADORES  | CCI14=%.2f | CMO14=%.2f | Stoch=%.2f/%.2f | WR=%.2f | ROC=%.2f",
            snapshot.cci_14,
            snapshot.cmo_14,
            snapshot.stoch_k,
            snapshot.stoch_d,
            snapshot.williams_r_14,
            snapshot.roc_14,
        )
        orch.logger.info(
            "  VOLATILIDADE | BB%%B=%.2f | BBw=%.2f%% | KC%%B=%.2f | ATR14=%.2f | ATR50=%.2f | ratio=%.2f",
            snapshot.bollinger_pct_b,
            snapshot.bollinger_width * 100.0,
            snapshot.keltner_pct_b,
            snapshot.atr_14,
            snapshot.atr_50,
            snapshot.atr_ratio,
        )
        orch.logger.info(
            "  ESTRUTURA    | Hurst=%.2f | Z=%.2f | body=%.2fATR | wick=%.2f/%.2f | dir=%+.2f | streak=%+.2f",
            snapshot.hurst_14,
            snapshot.session_zscore,
            snapshot.body_atr,
            snapshot.upper_wick_atr,
            snapshot.lower_wick_atr,
            snapshot.candle_dir,
            snapshot.streak,
        )
        orch.logger.info(
            "LAYA_DECISION | %s M5 | dir=%s | P(CALL)=%.2f%% | P(PUT)=%.2f%% | prob=%.2f | conf=%.2f",
            decision.symbol,
            decision.direction.name,
            decision.prob_call * 100.0,
            decision.prob_put * 100.0,
            decision.probability,
            decision.confidence,
        )
        orch.logger.info(
            "  MODELO       | lat=%.2fms | stake=$%.2f (%.2f%%) | banca=$%.2f | model=%s",
            decision.latency_ms,
            stake,
            stake_pct * 100.0,
            balance,
            decision.model_version,
        )
    finally:
        restore_log_context(prev_cid, prev_sym)
