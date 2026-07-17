"""Worker assincrono de refresh da matriz de correlacao cruzada."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.infrastructure.market.timescale_correlation_reader import (
    correlation_matrix_to_cache,
    fetch_correlation_matrix,
)


logger = logging.getLogger("AETH")


class _CorrelationWorkerState:
    """Estado do worker de correlacao."""

    def __init__(self) -> None:
        self.task: asyncio.Task | None = None


_state = _CorrelationWorkerState()


def _triton_cfg(config: dict) -> dict:
    """Extrai configuracao de correlacao do bloco infra."""
    infra = config.get("infra") if isinstance(config, dict) else {}
    chunk = infra.get("triton") if isinstance(infra, dict) else {}
    return chunk if isinstance(chunk, dict) else {}


async def refresh_correlation_cache(orch: Any) -> None:
    """Atualiza matriz de correlacao no Redis a partir do TimescaleDB."""
    infra = getattr(orch, "infra", None)
    if infra is None or not infra.enabled:
        return
    cfg = _triton_cfg(orch.config)
    ts_cfg = orch.config.get("infra", {}).get("timescale", {})
    dsn = str(ts_cfg.get("dsn", "postgresql://aether:aether@localhost:5432/aether"))
    data_cfg = orch.config.get("data_handler", {}) or {}
    granularity = int(data_cfg.get("granularity", 60))
    bars = int(cfg.get("correlation_bars", 120))
    symbols = [str(s) for s in getattr(orch, "symbols", [])]
    if not symbols:
        return
    try:
        matrix = await fetch_correlation_matrix(dsn, symbols, granularity=granularity, bars=bars)
        store = getattr(orch, "state_store", None)
        setter = getattr(store, "set_string", None) if store is not None else None
        if callable(setter):
            await setter("corr_matrix", correlation_matrix_to_cache(matrix))
        orch._corr_matrix_cache = matrix
        logger.debug("CORR: matriz atualizada para %d simbolos", len(symbols))
    except Exception as exc:
        logger.warning("CORR: falha ao atualizar matriz: %s", exc)


async def _correlation_worker_loop(orch: Any) -> None:
    """Loop periodico de refresh da matriz de correlacao."""
    cfg = _triton_cfg(orch.config)
    every = max(1, int(cfg.get("correlation_refresh_cycles", 5)))
    while getattr(orch, "running", False):
        await refresh_correlation_cache(orch)
        for _ in range(every):
            if not getattr(orch, "running", False):
                return
            await asyncio.sleep(float(orch.config.get("orchestrator", {}).get("cycle_interval_seconds", 300)))


def start_correlation_worker(orch: Any) -> None:
    """Inicia task asyncio de correlacao em background."""
    if _state.task is not None and not _state.task.done():
        return
    try:
        loop = asyncio.get_running_loop()
        _state.task = loop.create_task(_correlation_worker_loop(orch))
    except RuntimeError:
        return


def stop_correlation_worker() -> None:
    """Cancela worker de correlacao se ativo."""
    if _state.task is not None and not _state.task.done():
        _state.task.cancel()
    _state.task = None
