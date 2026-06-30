"""Consulta metadados HTTP de modelos no NVIDIA Triton Inference Server."""

from __future__ import annotations

import asyncio
from typing import Any

from src.infrastructure.inference.triton_http import get_triton_model_metadata, triton_http_base_url
from src.infrastructure.inference.triton_inference_client import triton_enabled, triton_http_url


async def fetch_triton_model_metadata_async(config: dict, model_name: str) -> dict[str, Any]:
    """Busca metadados do modelo Triton em thread de I/O."""
    if not triton_enabled(config):
        raise RuntimeError("Triton desabilitado na configuracao")
    http_base = triton_http_base_url(triton_http_url(config))
    return await asyncio.to_thread(get_triton_model_metadata, http_base, str(model_name))


def parse_triton_input_dims(model_payload: dict[str, Any]) -> tuple[int, int | None]:
    """Extrai feature_dim (shape[2]) e lookback opcional (shape[1]) do payload Triton."""
    inputs = model_payload.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise RuntimeError("Triton metadata sem bloco inputs")
    first = inputs[0]
    if not isinstance(first, dict):
        raise RuntimeError("Triton metadata inputs[0] invalido")
    shape = first.get("shape")
    if not isinstance(shape, list) or len(shape) < 3:
        raise RuntimeError(f"Triton input shape invalido: {shape!r}")
    lookback_raw = int(shape[1])
    feature_dim = int(shape[2])
    if feature_dim <= 0:
        raise RuntimeError(f"Triton feature_dim invalido: {feature_dim}")
    lookback = lookback_raw if lookback_raw > 0 else None
    return feature_dim, lookback
