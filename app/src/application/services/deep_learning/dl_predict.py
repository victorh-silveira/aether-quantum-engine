"""Predicao e gating de execucao por simbolo no bridge Deep Learning."""

import logging
from typing import Any

import numpy as np

from src.application.services.deep_learning.dl_bridge_helpers import build_decision_entry
from src.application.services.deep_learning.dl_calibration import CalibratorState, calibrate_trade_score
from src.application.services.deep_learning.dl_feature_build import precompute_price_series
from src.application.services.deep_learning.dl_gating import (
    check_indicator_gating_bounds,
    gating_block_reason,
    resolve_confidence_thresholds,
    resolve_edge,
)
from src.application.services.deep_learning.dl_outcomes import blended_val_accuracy
from src.application.services.deep_learning.dl_symbol_runtime import guard_symbol_model
from src.application.services.deep_learning.model import predict_next_direction
from src.domain.models.trade import TradeDirection


logger = logging.getLogger("AETH")


def _calculate_trend_direction(prices, exec_cfg: dict) -> tuple[TradeDirection, str, int]:
    """Calcula a direcao da tendencia usando SMA ou EMA com periodo configurado."""
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
        trend_dir = TradeDirection.CALL if trend_val >= prev_trend_val else TradeDirection.PUT
    else:
        last_val = close_prices[-1] if len(close_prices) > 0 else 0.0
        trend_dir = TradeDirection.CALL if last_val >= trend_val else TradeDirection.PUT

    trend_type = "EMA" if trend_use_ema else "SMA"
    return trend_dir, trend_type, trend_period


def predict_symbol_decision(
    orch,
    symbol: str,
    model,
    prices,
    norm_stats,
    runtime: dict,
    params: dict[str, Any],
    train_loss: float | None,
    *,
    recovery_active: bool,
    granularity: int = 60,
    open_=None,
    high=None,
    low=None,
    micro=None,
) -> dict:
    """Gera predicao e gating de execucao com threshold de confianca."""
    _ = recovery_active
    val_accuracy = blended_val_accuracy(
        orch,
        symbol,
        float(runtime["val_accuracy"]),
        live_weight=float(params.get("val_acc_live_blend", 0.35)),
    )
    call_threshold, put_threshold = resolve_confidence_thresholds(params)
    min_val_accuracy = float(params.get("min_val_accuracy", 0.53))
    min_edge = float(params.get("min_edge_execute", 0.0))
    try:
        gran = int(granularity or params.get("granularity", 60))
        calibrator = runtime.get("calibrator") or CalibratorState()
        with guard_symbol_model(runtime):
            direction, prob, raw_prob = predict_next_direction(
                model,
                prices,
                lookback=int(runtime.get("lookback", params["lookback"])),
                norm_stats=norm_stats,
                granularity=gran,
                symbol=str(symbol),
                open_=open_,
                high=high,
                low=low,
                micro=micro,
                implied_vol_bars=int(params.get("implied_vol_bars", 60)),
                call_threshold=call_threshold,
                put_threshold=put_threshold,
                calibrator=calibrator,
            )
        exec_cfg = orch.config.get("orchestrator", {}).get("execution", {}) if hasattr(orch, "config") else {}
        trend_dir, trend_type, trend_period = _calculate_trend_direction(prices, exec_cfg)

        if direction is None:
            mandatory = bool(exec_cfg.get("mandatory_trade_each_cycle", False))
            if mandatory:
                raw = float(raw_prob)
                side_score = calibrate_trade_score(
                    raw_prob,
                    val_accuracy,
                    calibrator,
                    deploy_ok=runtime.get("deploy_ok", True),
                    is_put=trend_dir == TradeDirection.PUT,
                )
                edge = resolve_edge(raw_prob)
                entry = build_decision_entry(
                    trend_dir,
                    raw,
                    execute=True,
                    val_accuracy=val_accuracy,
                    edge=edge,
                    train_loss=train_loss,
                    raw_prob=raw_prob,
                    trade_score=side_score,
                    contract_duration=int(params.get("contract_duration", 60)),
                )
                entry["metrics"]["gate_reason"] = None
                entry["metrics"]["trend_fallback"] = True
                entry["metrics"]["trend_direction"] = trend_dir.name
                entry["metrics"]["llm_note"] += f" (Trend Fallback: {trend_type}-{trend_period} {trend_dir.name})"
                return entry

            raw = float(raw_prob)
            weak_dir = TradeDirection.CALL if raw > 0.5 else TradeDirection.PUT
            side_score = max(raw, 1.0 - raw)
            entry = build_decision_entry(
                weak_dir,
                raw,
                execute=False,
                val_accuracy=val_accuracy,
                edge=resolve_edge(raw_prob),
                train_loss=train_loss,
                raw_prob=raw_prob,
                trade_score=side_score,
                contract_duration=int(params.get("contract_duration", 60)),
            )
            entry["metrics"]["gate_reason"] = "confidence"
            entry["metrics"]["trend_direction"] = trend_dir.name
            return entry
        block = gating_block_reason(
            raw_prob,
            val_accuracy,
            min_val_accuracy=min_val_accuracy,
            call_threshold=call_threshold,
            put_threshold=put_threshold,
            min_edge=min_edge,
        )
        if block is None:
            indicator_cfg = params.get("indicator_gating", {})
            if indicator_cfg.get("enabled", False):
                series = precompute_price_series(
                    prices,
                    granularity=gran,
                    symbol=str(symbol),
                    open_=open_,
                    high=high,
                    low=low,
                    micro=micro,
                    implied_vol_bars=int(params.get("implied_vol_bars", 60)),
                )
                indicators = {
                    "hurst": float(series["hurst"][-1]),
                    "adx": float(series["adx"][-1]),
                    "vol_ratio_short_long": float(series["vol_ratio_short_long"][-1]),
                    "cmo": float(series["cmo"][-1]),
                    "keltner_pct_b": float(series["keltner_pct_b"][-1]),
                }
                block = check_indicator_gating_bounds(indicators, indicator_cfg)
        execute = block is None
        edge = resolve_edge(raw_prob)
        side_score = calibrate_trade_score(
            raw_prob,
            val_accuracy,
            calibrator,
            deploy_ok=runtime.get("deploy_ok", True),
            is_put=direction == TradeDirection.PUT,
        )
        entry = build_decision_entry(
            direction,
            prob,
            execute=execute,
            val_accuracy=val_accuracy,
            edge=edge,
            train_loss=train_loss,
            trade_score=side_score,
            raw_prob=raw_prob,
            val_brier=float(runtime.get("val_brier", 1.0)),
            val_ece=float(runtime.get("val_ece", 1.0)),
            contract_duration=int(params.get("contract_duration", 60)),
        )
        entry["metrics"]["gate_reason"] = block
        entry["metrics"]["trend_direction"] = trend_dir.name
        return entry
    except Exception as e:
        logger.error("DL: Falha na predicao para %s: %s", symbol, e)
        entry = build_decision_entry(
            None,
            0.0,
            execute=False,
            val_accuracy=val_accuracy,
            edge=0.0,
            train_loss=train_loss,
            contract_duration=int(params.get("contract_duration", 60)),
        )
        entry["metrics"]["gate_reason"] = "predict_error"
        return entry
