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
from src.infrastructure.inference.triton_http import (
    load_triton_models_sequential,
    post_triton_repository_reload,
    triton_model_ready,
    wait_triton_models_ready,
)


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


def _triton_wait_settings(config: dict) -> tuple[float, float]:
    """Resolve timeout e intervalo de poll para modelos ficarem ready."""
    infra = config.get("infra") if isinstance(config, dict) else {}
    chunk = infra.get("triton") if isinstance(infra, dict) else {}
    if not isinstance(chunk, dict):
        return 25.0, 0.5
    timeout = float(chunk.get("wait_ready_seconds", 25.0))
    poll = float(chunk.get("poll_ready_seconds", 0.5))
    return timeout, poll


def triton_infer_timeout_seconds(config: dict, *, bootstrap: bool = False) -> float:
    """Resolve deadline gRPC do Triton para ciclo M5 ou probe de bootstrap."""
    infra = config.get("infra") if isinstance(config, dict) else {}
    chunk = infra.get("triton") if isinstance(infra, dict) else {}
    if not isinstance(chunk, dict):
        return 5.0 if bootstrap else 0.85
    if bootstrap:
        return float(chunk.get("bootstrap_infer_timeout_seconds", 5.0))
    return float(chunk.get("infer_timeout_seconds", 0.85))


async def reload_triton_repository(config: dict, model_names: list[str] | None = None) -> bool:
    """Recarrega modelos via API explicita apos sincronizar artefatos no disco."""
    symbols = [str(name) for name in (model_names or []) if str(name)]
    if not symbols:
        if not triton_enabled(config):
            return False
        try:
            models = await asyncio.to_thread(post_triton_repository_reload, triton_http_url(config))
        except (urllib.error.URLError, OSError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("TRITON: falha ao consultar repositorio (%s): %s", triton_http_url(config), exc)
            return False
        names = [str(item.get("name", "")) for item in models if isinstance(item, dict)]
        logger.debug("TRITON: indice repositorio | modelos=%s", ",".join(n for n in names if n) or "-")
        return True
    return await wait_triton_models_stable(config, symbols, repo_changed=True)


def _models_needing_load_over_load(
    http_url: str,
    symbols: list[str],
    *,
    repo_changed: bool,
    changed_models: list[str] | None,
) -> list[str]:
    """Seleciona modelos para /load: artefato novo ou ainda nao ready (sem /unload)."""
    if changed_models is not None:
        changed = {str(name) for name in changed_models if str(name)}
        candidates = [sym for sym in symbols if sym in changed] if repo_changed else []
    elif repo_changed:
        candidates = list(symbols)
    else:
        candidates = []
    pending_ready = [sym for sym in symbols if not triton_model_ready(http_url, sym)]
    ordered: list[str] = []
    for sym in symbols:
        if (sym in candidates or sym in pending_ready) and sym not in ordered:
            ordered.append(sym)
    return ordered


async def wait_triton_models_stable(
    config: dict,
    model_names: list[str],
    *,
    repo_changed: bool = True,
    changed_models: list[str] | None = None,
) -> bool:
    """MODE_EXPLICIT load-over-load: so /load nos modelos necessarios, nunca /unload."""
    if not triton_enabled(config):
        return False
    symbols = [str(name) for name in model_names if str(name)]
    if not symbols:
        return True
    http_url = triton_http_url(config)
    timeout, poll = _triton_wait_settings(config)
    if not repo_changed and not changed_models:
        ready = await asyncio.to_thread(
            wait_triton_models_ready,
            http_url,
            symbols,
            timeout_seconds=timeout,
            poll_interval_seconds=poll,
        )
        if ready:
            return True
    try:
        to_load = await asyncio.to_thread(
            _models_needing_load_over_load,
            http_url,
            symbols,
            repo_changed=repo_changed,
            changed_models=changed_models,
        )
        if to_load:
            loaded = await asyncio.to_thread(
                load_triton_models_sequential,
                http_url,
                to_load,
                wait_each_ready=True,
                timeout_seconds=timeout,
                poll_interval_seconds=poll,
            )
            logger.info("TRITON: load-over-load | modelos=%s", ",".join(loaded) or "-")
        else:
            logger.debug("TRITON: load-over-load omitido | modelos ja ready")
    except (urllib.error.URLError, OSError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("TRITON: falha no load-over-load (%s): %s", http_url, exc)
        return False
    return await asyncio.to_thread(
        wait_triton_models_ready,
        http_url,
        symbols,
        timeout_seconds=timeout,
        poll_interval_seconds=poll,
    )


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
