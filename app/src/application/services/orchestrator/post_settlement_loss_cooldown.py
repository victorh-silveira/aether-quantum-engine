"""Cooldownamento e bloqueio de cooldown tecnico apos LOSS consecutivas."""

import asyncio
import time
from typing import Any

from src.application.services.execution_runtime_config import resolve_post_loss_cooldown_config
from src.application.services.log_dedupe import log_info_if_changed
from src.application.services.regime_micro_freeze import SIGNAL_SUSPENDED


COOLDOWN_CYCLE_SUSPENDED = SIGNAL_SUSPENDED


def _exec_cfg_from_orch(orch: Any | None) -> dict[str, Any] | None:
    """Extrai orchestrator.execution do orquestrador quando disponivel."""
    if orch is None:
        return None
    cfg = getattr(orch, "config", None)
    if not isinstance(cfg, dict):
        return None
    orch_cfg = cfg.get("orchestrator")
    if not isinstance(orch_cfg, dict):
        return None
    execution = orch_cfg.get("execution")
    return execution if isinstance(execution, dict) else None


def _cooldown_cfg(orch: Any | None = None) -> dict[str, Any]:
    """Resolve ladder pos-LOSS (SSOT + override do orch)."""
    return resolve_post_loss_cooldown_config(_exec_cfg_from_orch(orch))


def post_loss_cooldown_delay_seconds(linear_losses: int, orch: Any | None = None) -> float:
    """Pausa tecnica por LIN: L1/L2=300s, L3=600s, L4+=900s (SSOT)."""
    cfg = _cooldown_cfg(orch)
    lin = int(linear_losses or 0)
    if lin < int(cfg["lin_min"]):
        return 0.0
    if lin <= 1:
        return float(cfg["delay_seconds_lin1"])
    if lin == 2:
        return float(cfg["delay_seconds_lin2"])
    if lin == 3:
        return float(cfg["delay_seconds_lin3"])
    return float(cfg["delay_seconds_lin4"])


def post_loss_cooldown_active(last_outcome: str, linear_losses: int, orch: Any | None = None) -> bool:
    """True se ultimo trade foi LOSS e linear >= lin_min do SSOT."""
    cfg = _cooldown_cfg(orch)
    return str(last_outcome or "").upper() == "LOSS" and int(linear_losses or 0) >= int(cfg["lin_min"])


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
    """Agenda pausa tecnica pos-LOSS conforme ladder SSOT."""
    rm = getattr(orch, "risk_manager", None)
    linear = int(getattr(rm, "consecutive_losses_linear", 0) or 0)
    outcome = getattr(orch, "_last_settlement_outcome", "")
    if not post_loss_cooldown_active(outcome, linear, orch):
        return 0.0
    delay = post_loss_cooldown_delay_seconds(linear, orch)
    prev = float(getattr(orch, "_cooldown_until", 0.0) or 0.0)
    orch._cooldown_until = max(prev, time.time() + delay)
    return delay


def log_trading_cycle_cooldown_skip(orch: Any) -> None:
    """Emite log deduplicado de cooldown pos-loss consecutivo."""
    logger = getattr(orch, "logger", None)
    if logger is None:
        return
    rem = orchestrator_cooldown_remaining(orch)
    cid = f"C{int(getattr(orch, '_active_cycle_id', 0) or 0):04d}"
    lin_min = int(_cooldown_cfg(orch)["lin_min"])
    log_info_if_changed(
        orch,
        logger,
        "loss_cooldown_skip",
        f"{rem:.0f}",
        "[%s] COOLDOWN || pausa tecnica pos-loss (LIN>=%d) | restante=%.0fs",
        cid,
        lin_min,
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
