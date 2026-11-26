"""Barreira cooperativa de persistencia de sessao do orquestrador."""

from __future__ import annotations

from typing import Any


def linear_reset_occurred(risk_manager: Any) -> bool:
    """Compatibilidade: reset linear desativado na arquitetura Laya continua."""
    _ = risk_manager
    return False


def consume_linear_reset_flag(orch: Any) -> bool:
    """Compatibilidade: retorna False sem recovery linear."""
    _ = orch
    return False


def session_persistence_write_active(orch: Any) -> bool:
    """True enquanto uma escrita de persistencia de sessao esta em andamento."""
    return getattr(orch, "_session_persistence_write_active", False) is True


def session_persistence_blocks_trading_cycle(orch: Any) -> bool:
    """True quando o ciclo deve aguardar liberacao da barreira de persistencia."""
    return session_persistence_write_active(orch)
