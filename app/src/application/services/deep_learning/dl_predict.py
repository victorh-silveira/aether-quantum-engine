"""Predicao por simbolo no bridge Deep Learning."""

import asyncio
import logging
from typing import Any

from src.application.services.deep_learning.dl_outcomes import blended_val_accuracy
from src.application.services.deep_learning.dl_predict_async import predict_symbol_decision_async
from src.application.services.deep_learning.dl_predict_build import (
    build_prediction_context,
    build_prediction_entry,
    eager_local_predict,
)


logger = logging.getLogger("AETH")


def predict_symbol_decision_sync(
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
    """Gera predicao DL sincrona local direta (sem usar event loop ou gRPC)."""
    val_accuracy = blended_val_accuracy(
        orch,
        symbol,
        float(runtime["val_accuracy"]),
        live_weight=float(params.get("val_acc_live_blend", 0.35)),
    )
    cycle_id = int(getattr(orch, "_active_cycle_id", 0) or 0)
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
    granularity: int = 60,
    open_=None,
    high=None,
    low=None,
    micro=None,
    force_local: bool = False,
) -> dict:
    """Gera predicao DL sync (local ou remota via loop se necessario)."""
    if force_local:
        return predict_symbol_decision_sync(
            orch,
            symbol,
            model,
            prices,
            norm_stats,
            runtime,
            params,
            train_loss,
            granularity=granularity,
            open_=open_,
            high=high,
            low=low,
            micro=micro,
        )
    return asyncio.run(
        predict_symbol_decision_async(
            orch,
            symbol,
            model,
            prices,
            norm_stats,
            runtime,
            params,
            train_loss,
            granularity=granularity,
            open_=open_,
            high=high,
            low=low,
            micro=micro,
            force_local=force_local,
        )
    )
