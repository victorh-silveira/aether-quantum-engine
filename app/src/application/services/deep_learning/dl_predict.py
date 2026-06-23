"""Predicao e gating de execucao por simbolo no bridge Deep Learning."""

import logging
from typing import Any

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
from src.application.services.deep_learning.dl_trend import calculate_trend_direction
from src.application.services.deep_learning.model import predict_next_direction
from src.domain.models.trade import TradeDirection


logger = logging.getLogger("AETH")


def _handle_neutral_decision(
    raw_prob: float,
    val_accuracy: float,
    train_loss: float | None,
    params: dict,
    indicators_data: dict,
    *,
    exhaustion_enabled: bool,
    rsi_lower: float,
    rsi_upper: float,
    keltner_lower: float,
    keltner_upper: float,
    trend_dir: TradeDirection,
    call_votes: int,
    put_votes: int,
) -> dict:
    """Retorna a decisao quando o modelo nao tem direcao definida (None)."""
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
    gate = "confidence"
    rsi = indicators_data.get("rsi", 0.5)
    keltner = indicators_data.get("keltner", 0.5)
    if exhaustion_enabled and (
        (weak_dir == TradeDirection.PUT and (rsi < rsi_lower or keltner < keltner_lower))
        or (weak_dir == TradeDirection.CALL and (rsi > rsi_upper or keltner > keltner_upper))
    ):
        gate = "exhaustion_conflict"
    elif bool(params.get("trend_alignment_required", False)) and weak_dir != trend_dir:
        gate = "trend_conflict"
    entry["metrics"]["gate_reason"] = gate
    entry["metrics"]["trend_direction"] = trend_dir.name
    entry["metrics"]["call_votes"] = call_votes
    entry["metrics"]["put_votes"] = put_votes
    entry["metrics"]["indicators"] = indicators_data
    return entry


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
        trend_dir, trend_type, trend_period, call_votes, put_votes = calculate_trend_direction(prices, series, exec_cfg)

        indicators_data = {
            "hurst": float(series["hurst"][-1]) if "hurst" in series and len(series["hurst"]) > 0 else 0.0,
            "adx": float(series["adx"][-1]) if "adx" in series and len(series["adx"]) > 0 else 0.0,
            "vol_ratio": float(series["vol_ratio_short_long"][-1])
            if "vol_ratio_short_long" in series and len(series["vol_ratio_short_long"]) > 0
            else 0.0,
            "cmo": float(series["cmo"][-1]) if "cmo" in series and len(series["cmo"]) > 0 else 0.0,
            "keltner": float(series["keltner_pct_b"][-1])
            if "keltner_pct_b" in series and len(series["keltner_pct_b"]) > 0
            else 0.0,
            "rsi": float(series["rsi"][-1]) if "rsi" in series and len(series["rsi"]) > 0 else 0.0,
            "macd": float(series["macd"][-1]) if "macd" in series and len(series["macd"]) > 0 else 0.0,
            "macd_sig": float(series["macd_signal"][-1])
            if "macd_signal" in series and len(series["macd_signal"]) > 0
            else 0.0,
            "di_diff": float(series["di_diff"][-1]) if "di_diff" in series and len(series["di_diff"]) > 0 else 0.0,
        }

        exhaustion_enabled = bool(params.get("exhaustion_filter_enabled", False))
        rsi_lower = float(params.get("exhaustion_rsi_lower", 0.28))
        rsi_upper = float(params.get("exhaustion_rsi_upper", 0.72))
        keltner_lower = float(params.get("exhaustion_keltner_lower", 0.0))
        keltner_upper = float(params.get("exhaustion_keltner_upper", 1.0))

        if direction is None:
            return _handle_neutral_decision(
                raw_prob,
                val_accuracy,
                train_loss,
                params,
                indicators_data,
                exhaustion_enabled=exhaustion_enabled,
                rsi_lower=rsi_lower,
                rsi_upper=rsi_upper,
                keltner_lower=keltner_lower,
                keltner_upper=keltner_upper,
                trend_dir=trend_dir,
                call_votes=call_votes,
                put_votes=put_votes,
            )
        block = gating_block_reason(
            raw_prob,
            val_accuracy,
            min_val_accuracy=min_val_accuracy,
            call_threshold=call_threshold,
            put_threshold=put_threshold,
            min_edge=min_edge,
        )
        rsi = indicators_data.get("rsi", 0.5)
        keltner = indicators_data.get("keltner", 0.5)
        if exhaustion_enabled and (
            (direction == TradeDirection.PUT and (rsi < rsi_lower or keltner < keltner_lower))
            or (direction == TradeDirection.CALL and (rsi > rsi_upper or keltner > keltner_upper))
        ):
            block = "exhaustion_conflict"
        elif bool(params.get("trend_alignment_required", False)) and direction != trend_dir:
            block = "trend_conflict"
        if block is None:
            indicator_cfg = params.get("indicator_gating", {})
            if indicator_cfg.get("enabled", False):
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
        entry["metrics"]["call_votes"] = call_votes
        entry["metrics"]["put_votes"] = put_votes
        entry["metrics"]["indicators"] = indicators_data
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
