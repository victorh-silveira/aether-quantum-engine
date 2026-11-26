"""Guardas de sincronia e cadencia M5 para operacao continua sem vetos ou skips."""

from __future__ import annotations

import time
from typing import Any

from src.application.services.orchestrator.api_maintenance_guard import api_maintenance_blocks_trading_cycle
from src.application.services.orchestrator.orchestrator_data_signature import (
    get_data_state_signature,
    m5_boundary_epoch,
    resolve_signature_boundary_seconds,
)
from src.application.services.orchestrator.session_persistence_barrier import session_persistence_blocks_trading_cycle


def cycle_cadence_seconds(orch: Any) -> int:
    """Retorna granularidade do ciclo M5 (300s) a partir da configuracao."""
    return int(resolve_signature_boundary_seconds(orch))


def mark_cycle_attempt_complete(orch: Any) -> None:
    """Atualiza timestamp da ultima tentativa de ciclo concluida."""
    orch._last_cluster_cycle_end = time.time()


def mark_boundary_cycle_evaluated(orch: Any) -> None:
    """Registra epoch de fronteira M5 avaliado para evitar reentrada na mesma vela."""
    orch._last_evaluated_boundary_epoch = m5_boundary_epoch(orch)


def commit_trading_cycle_data_signature(orch: Any) -> None:
    """Grava assinatura M5 corrente apos conclusao do ciclo."""
    sig = get_data_state_signature(orch)
    if sig:
        orch.last_data_signature = sig


def check_and_clear_boundary_cooldown(orch: Any) -> bool:
    """Limpa cooldown residual caso exista."""
    cooldown = float(getattr(orch, "_cooldown_until", 0.0) or 0.0)
    if cooldown > 0.0 and time.time() >= cooldown:
        orch._cooldown_until = 0.0
        return True
    return False


def trading_cycle_entry_allowed(orch: Any) -> bool:
    """Libera entrada continua quando ha nova vela M5 fechada e nenhum contrato ativo."""
    if not getattr(orch, "running", True) or getattr(orch, "is_trading", False):
        return False
    if getattr(orch, "_reconciliation_pending", False):
        return False
    if api_maintenance_blocks_trading_cycle(orch) or session_persistence_blocks_trading_cycle(orch):
        return False
    if getattr(orch.state, "active_contracts", None):
        return False
    boundary = m5_boundary_epoch(orch)
    if int(getattr(orch, "_last_evaluated_boundary_epoch", 0) or 0) == boundary:
        return False
    sig = get_data_state_signature(orch)
    return not (sig and sig == getattr(orch, "last_data_signature", ""))
