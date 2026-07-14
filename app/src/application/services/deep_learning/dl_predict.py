"""Predicao por simbolo no bridge Deep Learning."""

import asyncio
from typing import Any

from src.application.services.deep_learning.dl_outcomes import blended_val_accuracy
from src.application.services.deep_learning.dl_predict_async import predict_symbol_decision_async
from src.application.services.deep_learning.dl_predict_build import (
    build_prediction_context,
    build_prediction_entry,
    eager_local_predict,
)
from src.application.services.deep_learning.dl_predict_cache import (
    resolve_cached_prediction,
)
from src.application.services.orchestrator.orchestrator_data_signature import (
    at_signature_boundary,
)


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
    recovery_active: bool,
    granularity: int = 60,
    open_=None,
    high=None,
    low=None,
    micro=None,
) -> dict:
    """Gera predicao DL sincrona local direta (sem usar event loop ou gRPC)."""
    _ = recovery_active
    val_accuracy = blended_val_accuracy(
        orch,
        symbol,
        float(runtime["val_accuracy"]),
        live_weight=float(params.get("val_acc_live_blend", 0.35)),
    )
    on_boundary = at_signature_boundary(orch)
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
    cached = resolve_cached_prediction(orch, symbol, at_boundary=on_boundary)
    if cached is not None:
        return cached
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
    recovery_active: bool,
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
            recovery_active=recovery_active,
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
            recovery_active=recovery_active,
            granularity=granularity,
            open_=open_,
            high=high,
            low=low,
            micro=micro,
            force_local=force_local,
        )
    )
