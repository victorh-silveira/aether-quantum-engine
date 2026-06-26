"""Predicao por simbolo no bridge Deep Learning."""

import logging
from typing import Any

from src.application.services.deep_learning.dl_bridge_helpers import build_decision_entry
from src.application.services.deep_learning.dl_calibration import CalibratorState, calibrate_trade_score
from src.application.services.deep_learning.dl_feature_build import precompute_price_series
from src.application.services.deep_learning.dl_gating import resolve_confidence_thresholds, resolve_edge
from src.application.services.deep_learning.dl_outcomes import blended_val_accuracy
from src.application.services.deep_learning.dl_symbol_runtime import guard_symbol_model
from src.application.services.deep_learning.dl_trend import calculate_trend_direction
from src.application.services.deep_learning.model import predict_next_direction
from src.domain.models.trade import TradeDirection


logger = logging.getLogger("AETH")


def _infer_direction(raw_prob: float, direction: TradeDirection | None) -> TradeDirection:
    """Infere CALL ou PUT a partir de raw_prob quando direction e None."""
    if direction is not None:
        return direction
    return TradeDirection.CALL if float(raw_prob) > 0.5 else TradeDirection.PUT


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
    """Gera predicao DL com indicadores e tendencia para resolucao direcional."""
    _ = recovery_active
    val_accuracy = blended_val_accuracy(
        orch,
        symbol,
        float(runtime["val_accuracy"]),
        live_weight=float(params.get("val_acc_live_blend", 0.35)),
    )
    call_threshold, put_threshold = resolve_confidence_thresholds(params)
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

        resolved_dir = _infer_direction(raw_prob, direction)
        edge = resolve_edge(raw_prob)
        side_score = calibrate_trade_score(
            raw_prob,
            val_accuracy,
            calibrator,
            deploy_ok=runtime.get("deploy_ok", True),
            is_put=resolved_dir == TradeDirection.PUT,
        )
        entry = build_decision_entry(
            resolved_dir,
            prob,
            execute=True,
            val_accuracy=val_accuracy,
            edge=edge,
            train_loss=train_loss,
            trade_score=side_score,
            raw_prob=raw_prob,
            val_brier=float(runtime.get("val_brier", 1.0)),
            val_ece=float(runtime.get("val_ece", 1.0)),
            contract_duration=int(params.get("contract_duration", 60)),
        )
        entry["metrics"]["gate_reason"] = None
        entry["metrics"]["trend_direction"] = trend_dir.name
        entry["metrics"]["trend_type"] = trend_type
        entry["metrics"]["trend_period"] = trend_period
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
