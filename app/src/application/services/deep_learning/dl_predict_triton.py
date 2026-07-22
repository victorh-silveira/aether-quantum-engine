"""Predicao DL assincrona via Triton Inference Server."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.application.services.deep_learning.dl_calibration import apply_calibrator_stable
from src.application.services.deep_learning.model import predict_next_direction
from src.domain.models.trade import TradeDirection
from src.infrastructure.inference.triton_grpc_client import TritonInferenceTimeout
from src.infrastructure.inference.triton_inference_client import infer_symbol_async, triton_enabled
from src.infrastructure.inference.triton_tensor_builder import build_inference_tensor


logger = logging.getLogger("AETH")


def _require_triton_for_execution(config: dict | None) -> bool:
    """True quando timeout Triton deve falhar fechado sem fallback local."""
    cfg = config if isinstance(config, dict) else {}
    infra = cfg.get("infra", {}) if isinstance(cfg.get("infra"), dict) else {}
    triton = infra.get("triton", {}) if isinstance(infra.get("triton"), dict) else {}
    if "require_for_execution" in triton:
        return bool(triton.get("require_for_execution"))
    exec_cfg = cfg.get("orchestrator", {}).get("execution", {}) if isinstance(cfg.get("orchestrator"), dict) else {}
    if isinstance(exec_cfg, dict) and "require_triton_for_execution" in exec_cfg:
        return bool(exec_cfg.get("require_triton_for_execution"))
    return False


def _local_torchscript_predict(
    *,
    model: Any,
    prices: np.ndarray,
    lookback: int,
    norm_stats: Any,
    granularity: int,
    symbol: str,
    open_: np.ndarray | None,
    high: np.ndarray | None,
    low: np.ndarray | None,
    micro: dict[str, np.ndarray] | None,
    implied_vol_bars: int,
    call_th: float,
    put_th: float,
    calibrator: Any,
) -> tuple[TradeDirection | None, float, float]:
    """Executa predicao eager TorchScript local com calibracao."""
    direction, prob, raw_prob = predict_next_direction(
        model,
        prices,
        lookback,
        norm_stats=norm_stats,
        granularity=granularity,
        symbol=symbol,
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


def _resolve_direction(
    prob: float,
    raw_prob: float,
    call_th: float,
    put_th: float,
) -> tuple[TradeDirection | None, float, float]:
    """Mapeia probabilidade calibrada para direcao CALL, PUT ou neutro."""
    if prob + 1e-9 >= call_th:
        return TradeDirection.CALL, float(prob), float(raw_prob)
    if prob - 1e-9 <= put_th:
        return TradeDirection.PUT, float(prob), float(raw_prob)
    return None, float(prob), float(raw_prob)


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
    prebuilt_tensor: np.ndarray | None = None,
) -> tuple[TradeDirection | None, float, float]:
    """Retorna direcao, probabilidade calibrada e bruta via Triton ou eager local."""
    lookback = int(runtime.get("lookback", params["lookback"]))
    norm_stats = runtime["norm_stats"]
    calibrator = runtime.get("calibrator")
    call_th = float(call_threshold if call_threshold is not None else params.get("confidence_call_threshold", 0.75))
    put_th = float(put_threshold if put_threshold is not None else params.get("confidence_put_threshold", 0.25))
    implied_vol_bars = int(params.get("implied_vol_bars", 60))
    local_kwargs = {
        "model": runtime.get("model"),
        "prices": prices,
        "lookback": lookback,
        "norm_stats": norm_stats,
        "granularity": granularity,
        "symbol": str(symbol),
        "open_": open_,
        "high": high,
        "low": low,
        "micro": micro,
        "implied_vol_bars": implied_vol_bars,
        "call_th": call_th,
        "put_th": put_th,
        "calibrator": calibrator,
    }

    if not triton_enabled(orch.config):
        model = local_kwargs["model"]
        if model is None:
            return None, 0.5, 0.5
        return _local_torchscript_predict(**local_kwargs)

    if prebuilt_tensor is not None:
        tensor = np.asarray(prebuilt_tensor, dtype=np.float32)
    else:
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
    try:
        raw_prob = await infer_symbol_async(orch.config, str(symbol), tensor)
    except TritonInferenceTimeout:
        require_triton = _require_triton_for_execution(orch.config)
        if require_triton:
            logger.warning("TRITON_TIMEOUT_FAIL_CLOSED | sym=%s", symbol)
            return None, 0.5, 0.5
        logger.warning("TRITON_TIMEOUT_FALLBACK | sym=%s", symbol)
        model = local_kwargs["model"]
        if model is None:
            return None, 0.5, 0.5
        return _local_torchscript_predict(**local_kwargs)

    prob = apply_calibrator_stable(raw_prob, calibrator)
    return _resolve_direction(prob, raw_prob, call_th, put_th)
