"""Facade de inferencia Triton: delega ao cliente gRPC aio persistente."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.error
from typing import Any

import numpy as np

from src.infrastructure.inference.triton_grpc_client import (
    close_triton_grpc_client,
    get_triton_grpc_client,
)
from src.infrastructure.inference.triton_http import post_triton_repository_reload


logger = logging.getLogger("AETH")
_OUTPUT_NAME = "OUTPUT__0"


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


def triton_http_url(config: dict) -> str:
    """Resolve URL HTTP do Triton a partir da configuracao."""
    infra = config.get("infra") if isinstance(config, dict) else {}
    chunk = infra.get("triton") if isinstance(infra, dict) else {}
    if isinstance(chunk, dict) and chunk.get("http_url"):
        return str(chunk["http_url"]).rstrip("/")
    return os.getenv("AETHER_TRITON_HTTP", "http://localhost:8000").rstrip("/")


async def reload_triton_repository(config: dict) -> bool:
    """Recarrega modelos no Triton apos sincronizar artefatos no disco."""
    if not triton_enabled(config):
        return False
    http_url = triton_http_url(config)
    try:
        models = await asyncio.to_thread(post_triton_repository_reload, http_url)
    except (urllib.error.URLError, OSError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("TRITON: falha ao recarregar repositorio (%s): %s", http_url, exc)
        return False
    names = [str(item.get("name", "")) for item in models if isinstance(item, dict)]
    logger.debug("TRITON: repositorio recarregado | modelos=%s", ",".join(n for n in names if n) or "-")
    return True


async def get_triton_client(config: dict) -> Any:
    """Retorna cliente gRPC aio singleton para o endpoint configurado."""
    return await get_triton_grpc_client(triton_grpc_url(config))


async def close_triton_client() -> None:
    """Fecha cliente gRPC aio se aberto."""
    await close_triton_grpc_client()


def _parse_raw_output(result: Any) -> float:
    """Extrai probabilidade bruta do resultado Triton."""
    out = getattr(result, "as_numpy", lambda _n: None)(_OUTPUT_NAME)
    if out is None:
        return 0.5
    flat = np.asarray(out, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return 0.5
    val = float(flat[0])
    if not np.isfinite(val):
        return 0.5
    return max(0.0, min(1.0, val))


async def infer_symbol_async(config: dict, symbol: str, tensor: np.ndarray) -> float:
    """Executa inferencia gRPC para um simbolo e retorna probabilidade bruta."""
    client = await get_triton_grpc_client(triton_grpc_url(config))
    return await client.infer_symbol(str(symbol), tensor)


async def infer_symbols_async(config: dict, tensors: dict[str, np.ndarray]) -> dict[str, float]:
    """Inferencia gRPC concorrente para multiplos simbolos."""
    client = await get_triton_grpc_client(triton_grpc_url(config))
    return await client.infer_symbols_concurrent(tensors)
