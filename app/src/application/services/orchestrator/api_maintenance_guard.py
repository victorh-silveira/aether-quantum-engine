"""Portao de hibernacao cooperativa durante manutencao ou reset de liquidez do broker."""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime, timedelta
from typing import Any

from src.application.services.regime_micro_freeze import SIGNAL_SUSPENDED


_API_MAINTENANCE_FALLBACK_SECONDS = 65.0
_API_MAINTENANCE_SIGNATURES = (
    "trading is not available",
    "market is closed",
    "market is presently closed",
)
_MAINTENANCE_WINDOW_RE = re.compile(
    r"from\s+(\d{1,2}:\d{2}:\d{2})\s+to\s+(\d{1,2}:\d{2}:\d{2})",
    re.IGNORECASE,
)
_API_GUARD_LOG_MESSAGE = (
    "[AETHER] API_GUARD | Hibernando motor devido a reset/manutencao de liquidez do broker. "
    "Aguardando liberacao do pool."
)
MAINTENANCE_CYCLE_SUSPENDED = SIGNAL_SUSPENDED


def is_api_maintenance_error(error: BaseException | str) -> bool:
    """True quando a mensagem indica janela de indisponibilidade do broker."""
    message = str(error).lower()
    return any(signature in message for signature in _API_MAINTENANCE_SIGNATURES)


def _parse_clock(value: str) -> tuple[int, int, int] | None:
    """Converte HH:MM:SS em componentes numericos ou None quando invalido."""
    parts = value.strip().split(":")
    if len(parts) != 3:
        return None
    try:
        hour, minute, second = (int(part) for part in parts)
    except ValueError:
        return None
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        return None
    return hour, minute, second


def api_maintenance_delay_seconds(
    error: BaseException | str,
    *,
    now: datetime | None = None,
) -> float:
    """Calcula duracao da hibernacao a partir da janela informada pelo broker."""
    message = str(error)
    match = _MAINTENANCE_WINDOW_RE.search(message)
    if match is None:
        return _API_MAINTENANCE_FALLBACK_SECONDS
    start_clock = _parse_clock(match.group(1))
    end_clock = _parse_clock(match.group(2))
    if start_clock is None or end_clock is None:
        return _API_MAINTENANCE_FALLBACK_SECONDS
    current = now if now is not None else datetime.now(UTC)
    start_hour, start_minute, start_second = start_clock
    end_hour, end_minute, end_second = end_clock
    start_at = current.replace(
        hour=start_hour,
        minute=start_minute,
        second=start_second,
        microsecond=0,
    )
    end_at = current.replace(
        hour=end_hour,
        minute=end_minute,
        second=end_second,
        microsecond=0,
    )
    if end_at <= start_at:
        end_at += timedelta(days=1)
    if current > end_at:
        return _API_MAINTENANCE_FALLBACK_SECONDS
    delay = (end_at - current).total_seconds()
    if delay <= 0.0:
        return _API_MAINTENANCE_FALLBACK_SECONDS
    return float(delay)


def orchestrator_api_maintenance_until(orch: Any) -> float:
    """Retorna timestamp limite de hibernacao API no relogio do loop."""
    return float(getattr(orch, "_api_maintenance_until", 0.0))


def orchestrator_api_maintenance_active(orch: Any, *, now: float | None = None) -> bool:
    """True enquanto o motor permanece em hibernacao por manutencao do broker."""
    deadline = orchestrator_api_maintenance_until(orch)
    if deadline <= 0.0:
        return False
    current = now if now is not None else asyncio.get_running_loop().time()
    return current < deadline


def orchestrator_api_maintenance_remaining(orch: Any, *, now: float | None = None) -> float:
    """Segundos restantes de hibernacao API; zero quando inativo."""
    deadline = orchestrator_api_maintenance_until(orch)
    if deadline <= 0.0:
        return 0.0
    current = now if now is not None else asyncio.get_running_loop().time()
    return max(0.0, deadline - current)


def schedule_api_maintenance_hibernation(orch: Any, error: BaseException | str) -> float:
    """Registra barreira temporal de manutencao sem bloquear a corrotina."""
    if not is_api_maintenance_error(error):
        return 0.0
    delay = api_maintenance_delay_seconds(error)
    loop = asyncio.get_running_loop()
    orch._api_maintenance_until = loop.time() + delay
    return delay


def handle_broker_maintenance_error(orch: Any, error: BaseException | str) -> bool:
    """Agenda hibernacao quando o erro corresponde a manutencao do broker."""
    if not is_api_maintenance_error(error):
        return False
    schedule_api_maintenance_hibernation(orch, error)
    return True


def log_api_maintenance_hibernation(orch: Any) -> None:
    """Emite log deduplicado quando o ciclo e suspenso por manutencao do broker."""
    deadline = orchestrator_api_maintenance_until(orch)
    if deadline <= 0.0:
        return
    logged_until = float(getattr(orch, "_api_maintenance_logged_until", 0.0))
    if deadline <= logged_until:
        return
    orch._api_maintenance_logged_until = deadline
    orch.logger.info(_API_GUARD_LOG_MESSAGE)


def api_maintenance_blocks_trading_cycle(orch: Any) -> bool:
    """True quando o ciclo deve ser suspenso sem coletar TCN ou instanciar risco."""
    if not orchestrator_api_maintenance_active(orch):
        return False
    log_api_maintenance_hibernation(orch)
    return True
