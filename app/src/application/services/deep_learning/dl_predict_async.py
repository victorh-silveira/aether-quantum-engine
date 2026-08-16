"""Predicao DL assincrona por simbolo."""

from __future__ import annotations

import logging
from typing import Any

from src.application.services.deep_learning.dl_bridge_helpers import build_decision_entry
from src.application.services.deep_learning.dl_outcomes import blended_val_accuracy
from src.application.services.deep_learning.dl_predict_build import (
    build_prediction_context,
    build_prediction_entry,
    eager_local_predict,
)


logger = logging.getLogger("AETH")


async def predict_symbol_decision_async(
    orch,
    symbol: str,
    model,
    prices,
    norm_stats,
    runtime: dict,
    params: dict[str, Any],
    train_loss: float | None,
    *,
    granularity: int = 60,
    open_=None,
    high=None,
    low=None,
    micro=None,
) -> dict:
    """Gera predicao DL assincrona via eager local (CUDA/CPU)."""
    val_accuracy = blended_val_accuracy(
        orch,
        symbol,
        float(runtime["val_accuracy"]),
        live_weight=float(params.get("val_acc_live_blend", 0.35)),
    )
    cycle_id = int(getattr(orch, "_active_cycle_id", 0) or 0)
    try:
        ctx = build_prediction_context(
            orch,
            symbol,
            prices,
            runtime,
            params,
            granularity=granularity,
            open_=open_,
            high=high,
            low=low,
            micro=micro,
        )
        logger.debug("DL: predict fresh cycle=%d symbol=%s", cycle_id, symbol)
        direction, prob, raw_prob = eager_local_predict(
            ctx,
            model=model,
            prices=prices,
            norm_stats=norm_stats,
            runtime=runtime,
            params=params,
            symbol=symbol,
            open_=open_,
            high=high,
            low=low,
            micro=micro,
        )
        return build_prediction_entry(
            orch,
            symbol,
            prices,
            ctx["series"],
            runtime,
            params,
            train_loss,
            direction=direction,
            prob=float(prob),
            raw_prob=float(raw_prob),
            dynamic=ctx["dynamic"],
            dynamic_cfg=ctx["dynamic_cfg"],
            call_threshold=float(ctx["call_threshold"]),
            put_threshold=float(ctx["put_threshold"]),
            exec_cfg=ctx["exec_cfg"],
            val_accuracy=val_accuracy,
        )
    except Exception as e:
        logger.error("DL: Falha na predicao para %s: %s", symbol, e)
        entry = build_decision_entry(
            None,
            0.0,
            execute=False,
            val_accuracy=val_accuracy,
            edge=0.0,
            train_loss=train_loss,
            contract_duration=int(params.get("contract_duration", 120)),
        )
        entry["metrics"]["gate_reason"] = "predict_error"
        return entry
