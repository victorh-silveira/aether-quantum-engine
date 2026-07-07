"""Portao de aquecimento estatistico do TickBuffer apos reconexao do WebSocket."""

from __future__ import annotations

import asyncio
from typing import Any

from src.application.services.meta_direction_flip import SIGNAL_SUSPENDED


STREAM_WARM_UP_DELAY_SECONDS = 45.0
_WARM_UP_GUARD_LOG_MESSAGE = (
    "[AETHER] WARM_UP_GUARD | Aguardando estabilizacao do TickBuffer pos-reconexao. Coletando fluxo micro real."
)
WARM_UP_CYCLE_SUSPENDED = SIGNAL_SUSPENDED


def resolve_stream_warm_up_delay_seconds(config: dict[str, Any]) -> float:
    """Resolve duracao do aquecimento micro pos-reconexao em segundos."""
    chunk = config.get("orchestrator") if isinstance(config, dict) else {}
    orchestrator_cfg = chunk if isinstance(chunk, dict) else {}
    raw = orchestrator_cfg.get("stream_warm_up_delay_seconds", STREAM_WARM_UP_DELAY_SECONDS)
    return max(0.0, float(raw))


def stream_warm_up_deadline(orch: Any) -> float:
    """Retorna timestamp limite de aquecimento no relogio monotonico do loop."""
    return float(getattr(orch, "_stream_warmed_up_at", 0.0))


def stream_warm_up_active(orch: Any, *, now: float | None = None) -> bool:
    """True enquanto o buffer micro ainda nao atingiu estabilidade pos-reconexao."""
    deadline = stream_warm_up_deadline(orch)
    if deadline <= 0.0:
        return False
    current = now if now is not None else asyncio.get_running_loop().time()
    return current < deadline


def stream_warm_up_remaining(orch: Any, *, now: float | None = None) -> float:
    """Segundos restantes de aquecimento; zero quando inativo."""
    deadline = stream_warm_up_deadline(orch)
    if deadline <= 0.0:
        return 0.0
    current = now if now is not None else asyncio.get_running_loop().time()
    return max(0.0, deadline - current)


def schedule_stream_warm_up_barrier(orch: Any) -> float:
    """Registra barreira temporal de aquecimento apos restauracao do WebSocket."""
    delay = resolve_stream_warm_up_delay_seconds(getattr(orch, "config", {}) or {})
    loop = asyncio.get_running_loop()
    orch._stream_warmed_up_at = loop.time() + delay
    orch._warm_up_logged_until = 0.0
    return delay


def log_warm_up_guard_suspension(orch: Any) -> None:
    """Emite log deduplicado quando o ciclo e suspenso por aquecimento micro."""
    deadline = stream_warm_up_deadline(orch)
    if deadline <= 0.0:
        return
    logged_until = float(getattr(orch, "_warm_up_logged_until", 0.0))
    if deadline <= logged_until:
        return
    orch._warm_up_logged_until = deadline
    orch.logger.info(_WARM_UP_GUARD_LOG_MESSAGE)


def trading_cycle_warm_up_suspended(orch: Any) -> str | None:
    """Retorna SIGNAL_SUSPENDED quando o aquecimento micro ainda esta ativo."""
    if not stream_warm_up_active(orch):
        return None
    log_warm_up_guard_suspension(orch)
    return WARM_UP_CYCLE_SUSPENDED
