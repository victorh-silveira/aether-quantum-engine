"""Cliente gRPC assincrono para NVIDIA Triton Inference Server."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import numpy as np


try:
    import tritonclient.grpc.aio as grpc_aio
    from tritonclient.utils import InferenceServerException
except ImportError:
    grpc_aio = None
    InferenceServerException = Exception


logger = logging.getLogger("AETH")

_INPUT_NAME = "INPUT__0"
_OUTPUT_NAME = "OUTPUT__0"


class _TritonClientPool:
    """Pool singleton de cliente gRPC aio."""

    def __init__(self) -> None:
        self._client: Any | None = None
        self._client_url: str | None = None

    async def get(self, config: dict) -> Any:
        """Retorna cliente para o endpoint configurado."""
        if grpc_aio is None:
            raise RuntimeError("tritonclient[grpc] nao instalado")
        url = triton_grpc_url(config)
        if self._client is None or self._client_url != url:
            self._client = grpc_aio.InferenceServerClient(url=url, verbose=False)
            self._client_url = url
        return self._client

    async def close(self) -> None:
        """Fecha cliente se aberto."""
        if self._client is not None:
            await self._client.close()
            self._client = None
            self._client_url = None


_pool = _TritonClientPool()


def triton_enabled(config: dict) -> bool:
    """Indica se inferencia remota via Triton esta habilitada."""
    infra = config.get("infra") if isinstance(config, dict) else {}
    chunk = infra.get("triton") if isinstance(infra, dict) else {}
    return bool(chunk.get("enabled", False)) if isinstance(chunk, dict) else False


def triton_grpc_url(config: dict) -> str:
    """Resolve URL gRPC do Triton a partir da configuracao."""
    infra = config.get("infra") if isinstance(config, dict) else {}
    chunk = infra.get("triton") if isinstance(infra, dict) else {}
    if isinstance(chunk, dict) and chunk.get("grpc_url"):
        return str(chunk["grpc_url"])
    return os.getenv("AETHER_TRITON_GRPC", "localhost:8001")


async def get_triton_client(config: dict) -> Any:
    """Retorna cliente gRPC aio singleton para o endpoint configurado."""
    return await _pool.get(config)


async def close_triton_client() -> None:
    """Fecha cliente gRPC aio se aberto."""
    await _pool.close()


def _parse_raw_output(result: Any) -> float:
    """Extrai probabilidade bruta do resultado Triton."""
    out = result.as_numpy(_OUTPUT_NAME)
    if out is None:
        return 0.5
    flat = np.asarray(out, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return 0.5
    val = float(flat[0])
    if not np.isfinite(val):
        return 0.5
    return max(0.0, min(1.0, val))


async def infer_symbol_async(
    config: dict,
    symbol: str,
    tensor: np.ndarray,
) -> float:
    """Executa inferencia gRPC para um simbolo e retorna probabilidade bruta."""
    client = await get_triton_client(config)
    batch = np.asarray(tensor, dtype=np.float32)
    if batch.ndim == 2:
        batch = batch.reshape(1, batch.shape[0], batch.shape[1])
    inputs = [grpc_aio.InferInput(_INPUT_NAME, batch.shape, "FP32")]
    inputs[0].set_data_from_numpy(batch)
    outputs = [grpc_aio.InferRequestedOutput(_OUTPUT_NAME)]
    try:
        result = await client.infer(model_name=str(symbol), inputs=inputs, outputs=outputs)
        return _parse_raw_output(result)
    except InferenceServerException as exc:
        logger.error("TRITON: inferencia falhou para %s: %s", symbol, exc)
        raise


async def infer_symbols_async(
    config: dict,
    tensors: dict[str, np.ndarray],
) -> dict[str, float]:
    """Inferencia gRPC concorrente para multiplos simbolos."""

    async def _one(sym: str, arr: np.ndarray) -> tuple[str, float]:
        """Executa inferencia para um par simbolo/tensor."""
        prob = await infer_symbol_async(config, sym, arr)
        return sym, prob

    pairs = await asyncio.gather(*[_one(sym, arr) for sym, arr in tensors.items()])
    return dict(pairs)
