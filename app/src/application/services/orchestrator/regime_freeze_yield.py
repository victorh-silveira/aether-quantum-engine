"""Yield temporal quando o regime FREEZE suspende o ciclo de execucao."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from src.application.services.meta_direction_flip import SIGNAL_SUSPENDED


_REGIME_FREEZE_DEFAULT_YIELD_SECONDS = 15.0


def _entry_signal_suspended(entry: object) -> bool:
    """True quando o entry de decisao carrega SIGNAL_SUSPENDED."""
    if not isinstance(entry, dict):
        return False
    metrics = entry.get("metrics")
    return isinstance(metrics, dict) and str(metrics.get("signal_status") or "") == SIGNAL_SUSPENDED


def _entry_freeze_active(entry: object) -> bool:
    """True quando o entry sinaliza FREEZE ativo ou suspensao de sinal."""
    if not isinstance(entry, dict):
        return False
    if _entry_signal_suspended(entry):
        return True
    metrics = entry.get("metrics")
    if not isinstance(metrics, dict):
        return False
    action = str(metrics.get("regime_guard_action") or "")
    return action.startswith("FREEZE:")


def decisions_signal_suspended(decisions: dict) -> bool:
    """True quando qualquer simbolo do cluster reporta SIGNAL_SUSPENDED."""
    if not isinstance(decisions, dict):
        return False
    return any(_entry_signal_suspended(entry) for entry in decisions.values())


def cluster_freeze_active(decisions: dict) -> bool:
    """True quando qualquer simbolo do par exige suspensao global do cluster."""
    if not isinstance(decisions, dict):
        return False
    return any(_entry_freeze_active(entry) for entry in decisions.values())


def propagate_cluster_signal_suspended(decisions: dict) -> None:
    """Propaga SIGNAL_SUSPENDED para todos os simbolos quando o cluster congela."""
    if not isinstance(decisions, dict):
        return
    for entry in decisions.values():
        if not isinstance(entry, dict):
            continue
        metrics = entry.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}
            entry["metrics"] = metrics
        metrics["signal_status"] = SIGNAL_SUSPENDED


def regime_freeze_yield_seconds(orch: Any) -> float:
    """Calcula pausa ate a virada M1 ou fallback institucional de 15s."""
    epoch = int(getattr(orch, "_last_epoch", 0) or 0)
    if epoch > 0:
        next_boundary = ((epoch // 60) + 1) * 60
        remaining = float(next_boundary) - time.time()
        if remaining > 0.05:
            return remaining
    return _REGIME_FREEZE_DEFAULT_YIELD_SECONDS


async def _yield_freeze_delay(seconds: float) -> None:
    """Aguarda yield temporal sem acesso ao StateManager ou locks de sessao."""
    if seconds <= 0.0:
        return
    await asyncio.sleep(seconds)


async def await_regime_freeze_yield(orch: Any, decisions: dict) -> float:
    """Pausa o laço quando FREEZE suspende o ciclo, evitando hot loop."""
    if not getattr(orch, "running", True):
        return 0.0
    if not decisions_signal_suspended(decisions):
        return 0.0
    delay = regime_freeze_yield_seconds(orch)
    await _yield_freeze_delay(delay)
    return delay
