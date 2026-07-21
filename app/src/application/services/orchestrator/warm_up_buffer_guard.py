"""Portao de aquecimento estatistico do TickBuffer apos reconexao do WebSocket."""

from __future__ import annotations

import asyncio
from typing import Any

from src.application.services.force_trade_mode import force_trade_from_orch
from src.application.services.infra_timing_config import resolve_orchestrator_timing_config
from src.application.services.regime_micro_freeze import SIGNAL_SUSPENDED


def _timing(config: dict | None = None) -> dict:
    """Resolve knobs de timing do orchestrator."""
    orch = None
    if isinstance(config, dict):
        orch = config.get("orchestrator") if "orchestrator" in config else config
    return resolve_orchestrator_timing_config(orch if isinstance(orch, dict) else None)


_WARM_UP_GUARD_LOG_MESSAGE = (
    "[AETHER] WARM_UP_GUARD | Aguardando estabilizacao do TickBuffer pos-reconexao. Coletando fluxo micro real."
)
_WARM_UP_WAIT_LOG_MESSAGE = (
    "[AETHER] WARMUP_GUARD: Avaliando portao de aquecimento. Aguardando influxo de ticks reais da Deriv..."
)
_WARM_UP_WAIVER_LOG_MESSAGE = (
    "[AETHER] WARMUP_TIMEOUT: Influxo de ticks vivos estagnado. "
    "Forcando liberacao (Waiver) do loop mestre para evitar inanicao."
)
WARM_UP_CYCLE_SUSPENDED = SIGNAL_SUSPENDED


def resolve_stream_warm_up_delay_seconds(config: dict[str, Any]) -> float:
    """Resolve duracao do aquecimento micro pos-reconexao em segundos."""
    return max(0.0, float(_timing(config)["stream_warm_up_delay_seconds"]))


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
    orch._warm_up_waiver_applied = False
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
    if force_trade_from_orch(orch):
        return None
    if not stream_warm_up_active(orch):
        return None
    log_warm_up_guard_suspension(orch)
    return WARM_UP_CYCLE_SUSPENDED


def _tick_buffer_has_live_data(orch: Any, *, now: float | None = None) -> bool:
    """True quando o TickBuffer registrou atividade recente de ticks vivos."""
    stream = getattr(orch, "stream", None)
    buffer = getattr(stream, "tick_buffer", None) if stream is not None else None
    if buffer is None or not hasattr(buffer, "last_tick_monotonic"):
        return False
    last = float(buffer.last_tick_monotonic())
    if last <= 0.0:
        return False
    current = now if now is not None else asyncio.get_running_loop().time()
    return (current - last) < float(_timing()["warm_up_live_data_timeout_seconds"])


def apply_warm_up_initialization_waiver(orch: Any) -> None:
    """Libera o portao de aquecimento e sinaliza waiver de inicializacao."""
    orch._stream_warmed_up_at = 0.0
    orch._warm_up_waiver_applied = True
    orch._warm_up_logged_until = 0.0
    logger = getattr(orch, "logger", None)
    if logger is not None:
        logger.warning(_WARM_UP_WAIVER_LOG_MESSAGE)


async def await_stream_warm_up_gate(
    orch: Any,
    *,
    timeout: float = float(_timing()["warm_up_live_data_timeout_seconds"]),
) -> bool:
    """Aguarda aquecimento micro; apos timeout sem ticks vivos aplica waiver e segue."""
    if not stream_warm_up_active(orch):
        return True
    logger = getattr(orch, "logger", None)
    if logger is not None:
        logger.info(_WARM_UP_WAIT_LOG_MESSAGE)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(0.0, float(timeout))
    saw_live = False
    while stream_warm_up_active(orch):
        now = loop.time()
        if _tick_buffer_has_live_data(orch, now=now):
            saw_live = True
        if now >= deadline:
            if not saw_live:
                apply_warm_up_initialization_waiver(orch)
            return True
        await asyncio.sleep(0.1)
    return True
