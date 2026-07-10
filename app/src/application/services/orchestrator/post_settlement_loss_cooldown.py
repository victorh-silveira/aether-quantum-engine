"""Compatibilidade pos-LOSS sem barreira temporal (ciclos continuos)."""

from __future__ import annotations

from typing import Any

from src.application.services.meta_direction_flip import SIGNAL_SUSPENDED


COOLDOWN_CYCLE_SUSPENDED = SIGNAL_SUSPENDED


def post_loss_cooldown_delay_seconds(linear_losses: int) -> float:
    """Retorna zero: resfriamento pos-LOSS desativado."""
    _ = linear_losses
    return 0.0


def post_loss_cooldown_active(last_outcome: str, linear_losses: int) -> bool:
    """Retorna False: resfriamento pos-LOSS desativado."""
    _ = (last_outcome, linear_losses)
    return False


def orchestrator_cooldown_until(orch: Any) -> float:
    """Retorna zero: sem barreira temporal ativa."""
    _ = orch
    return 0.0


def orchestrator_cooldown_active(orch: Any, *, now: float | None = None) -> bool:
    """Retorna False: ciclos nunca suspensos por cooldown pos-LOSS."""
    _ = (orch, now)
    return False


def orchestrator_cooldown_remaining(orch: Any, *, now: float | None = None) -> float:
    """Retorna zero: sem tempo restante de resfriamento."""
    _ = (orch, now)
    return 0.0


def schedule_post_loss_cooldown(orch: Any) -> float:
    """Nao agenda resfriamento pos-LOSS."""
    _ = orch
    return 0.0


def log_trading_cycle_cooldown_skip(orch: Any) -> None:
    """Nao emite logs de resfriamento pos-LOSS."""
    _ = orch


def post_loss_cooldown_blocks_trading_cycle(orch: Any) -> bool:
    """Nunca bloqueia entrada de ciclo por cooldown pos-LOSS."""
    _ = orch
    return False


async def await_post_loss_cooldown(orch: Any) -> float:
    """Nao retém o loop apos liquidacao."""
    _ = orch
    return 0.0
