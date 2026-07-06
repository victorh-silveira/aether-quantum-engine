"""Gate de inercia temporal pos-LOSS antes do proximo ciclo de decisao."""

from __future__ import annotations

import asyncio
from typing import Any


POST_LOSS_COOLDOWN_BASE_SECONDS = 15.0
POST_LOSS_COOLDOWN_GROWTH = 1.35
POST_LOSS_COOLDOWN_LINEAR_MIN = 2


def post_loss_cooldown_delay_seconds(linear_losses: int) -> float:
    """Calcula atraso exponencial 15 * 1.35^n quando linear >= 2."""
    linear = max(0, int(linear_losses))
    if linear < POST_LOSS_COOLDOWN_LINEAR_MIN:
        return 0.0
    return POST_LOSS_COOLDOWN_BASE_SECONDS * (POST_LOSS_COOLDOWN_GROWTH**linear)


def post_loss_cooldown_active(last_outcome: str, linear_losses: int) -> bool:
    """Indica se o portao de cooling-down deve ser aplicado."""
    return str(last_outcome or "").upper() == "LOSS" and int(linear_losses) >= POST_LOSS_COOLDOWN_LINEAR_MIN


async def await_post_loss_cooldown(orch: Any) -> float:
    """Aguarda dissipacao micro pos-LOSS antes de liberar novo ciclo."""
    outcome = str(getattr(orch, "_last_settlement_outcome", "") or "")
    linear = int(getattr(orch.risk_manager, "consecutive_losses_linear", 0))
    if not post_loss_cooldown_active(outcome, linear):
        return 0.0
    delay = post_loss_cooldown_delay_seconds(linear)
    if delay <= 0.0 or not orch.running:
        return 0.0
    orch.logger.info(
        "CICLO: cooling-down %.1fs pos-LOSS linear=%d",
        delay,
        linear,
    )
    await asyncio.sleep(delay)
    return delay
