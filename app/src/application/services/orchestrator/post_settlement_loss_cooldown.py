"""Cooldownamento e bloqueio de cooldown tecnico apos LOSS consecutivas."""

import asyncio
import time
from typing import Any

from src.application.services.log_dedupe import log_info_if_changed
from src.application.services.regime_micro_freeze import SIGNAL_SUSPENDED


COOLDOWN_CYCLE_SUSPENDED = SIGNAL_SUSPENDED


def post_loss_cooldown_delay_seconds(linear_losses: int) -> float:
    """Pausa de 1 ciclo M5 (300s) quando linear_losses >= 2."""
    if int(linear_losses or 0) >= 2:
        return 300.0
    return 0.0


def post_loss_cooldown_active(last_outcome: str, linear_losses: int) -> bool:
    """True se ultimo trade foi LOSS e linear >= 2."""
    return str(last_outcome or "").upper() == "LOSS" and int(linear_losses or 0) >= 2


def orchestrator_cooldown_until(orch: Any) -> float:
    """Retorna timestamp limite de resfriamento."""
    return float(getattr(orch, "_cooldown_until", 0.0) or 0.0)


def orchestrator_cooldown_active(orch: Any, *, now: float | None = None) -> bool:
    """True enquanto o timestamp de cooldown estiver no futuro."""
    deadline = orchestrator_cooldown_until(orch)
    if deadline <= 0.0:
        return False
    current = float(now if now is not None else time.time())
    return current < deadline


def orchestrator_cooldown_remaining(orch: Any, *, now: float | None = None) -> float:
    """Tempo restante de resfriamento em segundos."""
    deadline = orchestrator_cooldown_until(orch)
    if deadline <= 0.0:
        return 0.0
    current = float(now if now is not None else time.time())
    return max(0.0, deadline - current)


def schedule_post_loss_cooldown(orch: Any) -> float:
    """Agenda pausa tecnica de 1 ciclo M5 (300s) se linear >= 2."""
    rm = getattr(orch, "risk_manager", None)
    linear = int(getattr(rm, "consecutive_losses_linear", 0) or 0)
    outcome = getattr(orch, "_last_settlement_outcome", "")
    if not post_loss_cooldown_active(outcome, linear):
        return 0.0
    delay = post_loss_cooldown_delay_seconds(linear)
    orch._cooldown_until = time.time() + delay
    return delay


def log_trading_cycle_cooldown_skip(orch: Any) -> None:
    """Emite log deduplicado de cooldown pos-loss consecutivo."""
    logger = getattr(orch, "logger", None)
    if logger is None:
        return
    rem = orchestrator_cooldown_remaining(orch)
    cid = f"C{int(getattr(orch, '_active_cycle_id', 0) or 0):04d}"
    log_info_if_changed(
        orch,
        logger,
        "loss_cooldown_skip",
        f"{rem:.0f}",
        "[%s] COOLDOWN || pausa tecnica pos-loss (LIN>=2) | restante=%.0fs",
        cid,
        rem,
    )


def post_loss_cooldown_blocks_trading_cycle(orch: Any) -> bool:
    """True se o motor estiver em resfriamento pos-loss."""
    active = orchestrator_cooldown_active(orch)
    if active:
        log_trading_cycle_cooldown_skip(orch)
    return active


async def await_post_loss_cooldown(orch: Any) -> float:
    """Aguarda resfriamento antes do proximo ciclo."""
    rem = orchestrator_cooldown_remaining(orch)
    if rem > 0.0:
        await asyncio.sleep(min(rem, 5.0))
    return rem
