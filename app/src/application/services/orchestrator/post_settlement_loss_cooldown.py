"""Gate de inercia temporal pos-LOSS via barreira de timestamp nao-bloqueante."""

from __future__ import annotations

import asyncio
from typing import Any

from src.application.services.meta_direction_flip import SIGNAL_SUSPENDED


POST_LOSS_COOLDOWN_BASE_SECONDS = 15.0
POST_LOSS_COOLDOWN_GROWTH = 1.35
POST_LOSS_COOLDOWN_LINEAR_MIN = 2
COOLDOWN_CYCLE_SUSPENDED = SIGNAL_SUSPENDED


def post_loss_cooldown_delay_seconds(linear_losses: int) -> float:
    """Calcula atraso exponencial 15 * 1.35^n quando linear >= 2."""
    linear = max(0, int(linear_losses))
    if linear < POST_LOSS_COOLDOWN_LINEAR_MIN:
        return 0.0
    return POST_LOSS_COOLDOWN_BASE_SECONDS * (POST_LOSS_COOLDOWN_GROWTH**linear)


def post_loss_cooldown_active(last_outcome: str, linear_losses: int) -> bool:
    """Indica se o portao de cooling-down deve ser aplicado."""
    return str(last_outcome or "").upper() == "LOSS" and int(linear_losses) >= POST_LOSS_COOLDOWN_LINEAR_MIN


def orchestrator_cooldown_until(orch: Any) -> float:
    """Retorna timestamp limite do resfriamento pos-LOSS no relogio do loop."""
    return float(getattr(orch, "_cooldown_until", 0.0))


def orchestrator_cooldown_active(orch: Any, *, now: float | None = None) -> bool:
    """True enquanto o motor permanece dentro do periodo de resfriamento pos-LOSS."""
    deadline = orchestrator_cooldown_until(orch)
    if deadline <= 0.0:
        return False
    current = now if now is not None else asyncio.get_running_loop().time()
    return current < deadline


def orchestrator_cooldown_remaining(orch: Any, *, now: float | None = None) -> float:
    """Segundos restantes de resfriamento pos-LOSS; zero quando inativo."""
    deadline = orchestrator_cooldown_until(orch)
    if deadline <= 0.0:
        return 0.0
    current = now if now is not None else asyncio.get_running_loop().time()
    return max(0.0, deadline - current)


def schedule_post_loss_cooldown(orch: Any) -> float:
    """Registra barreira temporal pos-LOSS sem bloquear a corrotina."""
    outcome = str(getattr(orch, "_last_settlement_outcome", "") or "")
    linear = int(getattr(orch.risk_manager, "consecutive_losses_linear", 0))
    if not post_loss_cooldown_active(outcome, linear):
        return 0.0
    delay = post_loss_cooldown_delay_seconds(linear)
    if delay <= 0.0 or not orch.running:
        return 0.0
    loop = asyncio.get_running_loop()
    orch._cooldown_until = loop.time() + delay
    orch.logger.info(
        "CICLO: cooling-down %.1fs pos-LOSS linear=%d",
        delay,
        linear,
    )
    return delay


def log_trading_cycle_cooldown_skip(orch: Any) -> None:
    """Emite log deduplicado quando o ciclo e suspenso pela barreira pos-LOSS."""
    deadline = orchestrator_cooldown_until(orch)
    if deadline <= 0.0:
        return
    logged_until = float(getattr(orch, "_cooldown_skip_logged_until", 0.0))
    if deadline <= logged_until:
        return
    orch._cooldown_skip_logged_until = deadline
    remaining = orchestrator_cooldown_remaining(orch)
    orch.logger.info(
        "CICLO: resfriamento pos-LOSS ativo (%.1fs restantes); ciclo suspenso",
        remaining,
    )


def post_loss_cooldown_blocks_trading_cycle(orch: Any) -> bool:
    """True quando o ciclo deve ser suspenso sem invocar resolver ou cluster."""
    if not orchestrator_cooldown_active(orch):
        return False
    log_trading_cycle_cooldown_skip(orch)
    return True


async def await_post_loss_cooldown(orch: Any) -> float:
    """Agenda barreira pos-LOSS sem reter o laço com sleep monolitico."""
    return schedule_post_loss_cooldown(orch)
