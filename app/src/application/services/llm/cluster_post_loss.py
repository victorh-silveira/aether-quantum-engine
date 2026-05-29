"""Pausa e veto de reentrada no cluster apos loss."""

from __future__ import annotations

from typing import Any

from src.domain.models.trade import TradeDirection


def record_cluster_loss(orch: Any, *, symbol: str, direction: TradeDirection | None) -> None:
    """Registra loss para pausa de cluster e bloqueio de setup repetido."""
    orch_cfg = orch.config.get("orchestrator", {}) if isinstance(getattr(orch, "config", None), dict) else {}
    pause_cycles = max(0, int(orch_cfg.get("cluster_pause_after_loss_cycles", 2)))
    orch._cluster_pause_cycles_remaining = pause_cycles
    orch._last_loss_symbol = str(symbol)
    orch._last_loss_direction = direction.name if isinstance(direction, TradeDirection) else ""


def cluster_post_loss_block_reason(
    orch: Any,
    *,
    target_sym: str,
    target_direction: TradeDirection | None,
) -> str | None:
    """Motivo de bloqueio por pausa pos-loss ou repeticao do ultimo loss; None se liberado."""
    if getattr(orch, "_cluster_pause_after_loss_active", False) is True:
        return "cluster_pause_after_loss"
    last_sym = str(getattr(orch, "_last_loss_symbol", "") or "")
    last_dir = str(getattr(orch, "_last_loss_direction", "") or "")
    if (
        last_sym
        and last_dir
        and target_sym == last_sym
        and isinstance(target_direction, TradeDirection)
        and target_direction.name == last_dir
    ):
        return "repeat_loss_setup"
    return None
