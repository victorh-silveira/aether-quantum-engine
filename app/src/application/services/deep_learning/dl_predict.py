"""Predicao e gating de execucao por simbolo no bridge Deep Learning."""

import logging
from typing import Any

from src.application.services.deep_learning.dl_bridge_helpers import build_decision_entry
from src.application.services.deep_learning.dl_gating import (
    gating_block_reason,
    resolve_confidence_thresholds,
    resolve_edge,
)
from src.application.services.deep_learning.dl_outcomes import blended_val_accuracy
from src.application.services.deep_learning.dl_symbol_runtime import guard_symbol_model
from src.application.services.deep_learning.model import predict_next_direction


logger = logging.getLogger("AETH")


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
            )
        if direction is None:
            entry = build_decision_entry(
                None,
                float(raw_prob),
                execute=False,
                val_accuracy=val_accuracy,
                edge=resolve_edge(raw_prob),
                train_loss=train_loss,
                raw_prob=raw_prob,
                trade_score=float(raw_prob),
                contract_duration=int(params.get("contract_duration", 60)),
            )
            entry["metrics"]["gate_reason"] = "confidence"
            return entry
        block = gating_block_reason(
            raw_prob,
            val_accuracy,
            min_val_accuracy=min_val_accuracy,
            call_threshold=call_threshold,
            put_threshold=put_threshold,
            min_edge=min_edge,
        )
        execute = block is None
        edge = resolve_edge(raw_prob)
        raw = float(raw_prob)
        side_score = max(raw, 1.0 - raw)
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
