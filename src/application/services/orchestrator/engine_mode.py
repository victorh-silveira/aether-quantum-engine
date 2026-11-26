"""Modo unico de operacao continua (execucao via Laya System 1)."""

from __future__ import annotations

from typing import Any


ENGINE_MODE_EXECUTE = "execute"


def resolve_engine_mode(config: dict[str, Any] | None = None) -> str:
    """Retorna sempre execute na arquitetura Laya continua."""
    _ = config
    return ENGINE_MODE_EXECUTE


def training_enabled(orch: Any = None) -> bool:
    """Modo treino dedicado removido na arquitetura Laya continua."""
    _ = orch
    return False
