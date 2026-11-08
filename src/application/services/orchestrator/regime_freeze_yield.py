"""Compatibilidade de regime FREEZE sem reter o loop (ciclos continuos)."""

from __future__ import annotations

from typing import Any

from src.application.services.regime_micro_freeze import SIGNAL_SUSPENDED


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


def cluster_collect_aborted(decisions: dict) -> bool:
    """Nao aborta coleta do cluster: trades continuam mesmo com FREEZE."""
    _ = decisions
    return False


def regime_freeze_yield_seconds(orch: Any) -> float:
    """Retorna zero: sem pausa por regime FREEZE."""
    _ = orch
    return 0.0


async def await_regime_freeze_yield(orch: Any, decisions: dict) -> float:
    """Nao retém o loop por regime FREEZE."""
    _ = (orch, decisions)
    return 0.0
