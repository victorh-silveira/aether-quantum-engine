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
from src.application.services.deep_learning.dl_predict_cache import (
    resolve_cached_prediction,
    store_prediction_cache,
)
from src.application.services.deep_learning.dl_predict_triton import predict_raw_prob_async
from src.application.services.orchestrator.orchestrator_data_signature import (
    at_signature_boundary,
    m5_boundary_epoch,
)
from src.infrastructure.inference.triton_inference_client import triton_enabled
from src.infrastructure.inference.triton_tensor_builder import (
    PartialInferenceHistoryError,
    build_inference_tensor,
    inference_tensor_fingerprint,
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
    force_local: bool = False,
) -> dict:
    """Gera predicao DL assincrona com inferencia Triton quando habilitada."""
    val_accuracy = blended_val_accuracy(
        orch,
        symbol,
        float(runtime["val_accuracy"]),
        live_weight=float(params.get("val_acc_live_blend", 0.35)),
    )
    lookback = int(runtime.get("lookback", params["lookback"]))
    on_boundary = at_signature_boundary(orch)
    boundary_epoch = m5_boundary_epoch(orch)
    cycle_id = int(getattr(orch, "_active_cycle_id", 0) or 0)
    use_triton = bool(triton_enabled(orch.config) and not force_local)
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
        tensor_fingerprint: bytes | None = None
        if use_triton:
            try:
                tensor = build_inference_tensor(
                    prices,
                    lookback,
                    runtime["norm_stats"],
                    granularity=int(ctx["gran"]),
                    symbol=str(symbol),
                    open_=open_,
                    high=high,
                    low=low,
                    micro=micro,
                    implied_vol_bars=int(params.get("implied_vol_bars", 60)),
                )
                tensor_fingerprint = inference_tensor_fingerprint(tensor)
            except PartialInferenceHistoryError:
                cached = resolve_cached_prediction(
                    orch,
                    symbol,
                    at_boundary=False,
                    boundary_epoch=boundary_epoch,
                    cycle_id=cycle_id,
                )
                if cached is not None:
                    return cached
                raise
            cached = resolve_cached_prediction(
                orch,
                symbol,
                at_boundary=on_boundary,
                tensor_fingerprint=tensor_fingerprint,
                boundary_epoch=boundary_epoch,
                cycle_id=cycle_id,
            )
            if cached is not None:
                return cached
            direction, prob, raw_prob = await predict_raw_prob_async(
                orch,
                symbol,
                prices,
                runtime,
                params,
                granularity=int(ctx["gran"]),
                open_=open_,
                high=high,
                low=low,
                micro=micro,
                call_threshold=float(ctx["call_threshold"]),
                put_threshold=float(ctx["put_threshold"]),
                prebuilt_tensor=tensor,
            )
        else:
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
        entry = build_prediction_entry(
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
        if tensor_fingerprint is not None:
            store_prediction_cache(
                orch,
                symbol,
                entry,
                tensor_fingerprint=tensor_fingerprint,
                boundary_epoch=boundary_epoch,
                cycle_id=cycle_id,
            )
        return entry
    except Exception as e:
        cached = resolve_cached_prediction(
            orch,
            symbol,
            at_boundary=False,
            boundary_epoch=boundary_epoch,
            cycle_id=cycle_id,
        )
        if cached is not None:
            return cached
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
