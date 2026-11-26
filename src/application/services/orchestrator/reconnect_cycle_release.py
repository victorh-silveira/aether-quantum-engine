"""Libera ciclos continuos de trading apos reconexao do WebSocket."""

from __future__ import annotations

import asyncio
from typing import Any

from src.application.services.orchestrator.warm_up_buffer_guard import resolve_stream_warm_up_delay_seconds


def schedule_post_reconnect_warm_up_barrier(orch: Any) -> float:
    """Registra barreira de aquecimento apos reconexao do WebSocket."""
    delay = resolve_stream_warm_up_delay_seconds(getattr(orch, "config", {}) or {})
    loop = asyncio.get_running_loop()
    orch._stream_warmed_up_at = loop.time() + delay
    orch._warm_up_logged_until = 0.0
    return delay


def release_trading_cycle_after_reconnect(orch: Any) -> None:
    """Invalida assinatura e epoch processado para retomar ciclos apos queda de rede."""
    orch.last_data_signature = ""
    orch._last_processed_epoch = 0
    orch._last_evaluated_boundary_epoch = 0
    delay = schedule_post_reconnect_warm_up_barrier(orch)
    logger = getattr(orch, "logger", None)
    if logger is not None:
        logger.info("RECOV: ciclo continuo liberado | warm_up=%.2fs", delay)
