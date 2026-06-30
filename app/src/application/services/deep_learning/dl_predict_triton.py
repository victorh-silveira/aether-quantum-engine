"""Predicao DL assincrona via Triton Inference Server."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.application.services.deep_learning.dl_calibration import apply_calibrator
from src.application.services.deep_learning.model import predict_next_direction
from src.domain.models.trade import TradeDirection
from src.infrastructure.inference.triton_inference_client import infer_symbol_async, triton_enabled
from src.infrastructure.inference.triton_tensor_builder import build_inference_tensor


logger = logging.getLogger("AETH")


async def predict_raw_prob_async(
    orch: Any,
    symbol: str,
    prices: np.ndarray,
    runtime: dict,
    params: dict[str, Any],
    *,
    granularity: int,
    open_: np.ndarray | None = None,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    micro: dict[str, np.ndarray] | None = None,
    call_threshold: float | None = None,
    put_threshold: float | None = None,
) -> tuple[TradeDirection | None, float, float]:
    """Retorna direcao, probabilidade calibrada e bruta via Triton ou eager local."""
    lookback = int(runtime.get("lookback", params["lookback"]))
    norm_stats = runtime["norm_stats"]
    calibrator = runtime.get("calibrator")
    call_th = float(call_threshold if call_threshold is not None else params.get("confidence_call_threshold", 0.75))
    put_th = float(put_threshold if put_threshold is not None else params.get("confidence_put_threshold", 0.25))
    implied_vol_bars = int(params.get("implied_vol_bars", 60))

    if triton_enabled(orch.config):
        tensor = build_inference_tensor(
            prices,
            lookback,
            norm_stats,
            granularity=granularity,
            symbol=str(symbol),
            open_=open_,
            high=high,
            low=low,
            micro=micro,
            implied_vol_bars=implied_vol_bars,
        )
        raw_prob = await infer_symbol_async(orch.config, str(symbol), tensor)
        prob = apply_calibrator(raw_prob, calibrator) if calibrator is not None else raw_prob
    else:
        model = runtime.get("model")
        if model is None:
            return None, 0.5, 0.5
        direction, prob, raw_prob = predict_next_direction(
            model,
            prices,
            lookback,
            norm_stats=norm_stats,
            granularity=granularity,
            symbol=str(symbol),
            open_=open_,
            high=high,
            low=low,
            micro=micro,
            implied_vol_bars=implied_vol_bars,
            call_threshold=call_th,
            put_threshold=put_th,
            calibrator=calibrator,
        )
        return direction, float(prob), float(raw_prob)

    if prob + 1e-9 >= call_th:
        return TradeDirection.CALL, float(prob), float(raw_prob)
    if prob - 1e-9 <= put_th:
        return TradeDirection.PUT, float(prob), float(raw_prob)
    return None, float(prob), float(raw_prob)
