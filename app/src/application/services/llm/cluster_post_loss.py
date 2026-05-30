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
    block_cycles = max(0, int(orch_cfg.get("cluster_repeat_loss_block_cycles", 0)))
    if block_cycles > 0:
        orch._repeat_loss_block_cycles_remaining = block_cycles


def tick_cluster_repeat_loss_cooldown(orch: Any) -> None:
    """Decrementa cooldown do veto repeat_loss_setup; zera ultimo loss ao expirar."""
    remaining = getattr(orch, "_repeat_loss_block_cycles_remaining", None)
    if remaining is None:
        return
    if int(remaining) <= 0:
        orch._last_loss_symbol = ""
        orch._last_loss_direction = ""
        orch._repeat_loss_block_cycles_remaining = None
        return
    orch._repeat_loss_block_cycles_remaining = int(remaining) - 1


def cluster_post_loss_block_reason(
    orch: Any,
    *,
    target_sym: str,
    target_direction: TradeDirection | None,
) -> str | None:
    """Motivo de bloqueio por pausa pos-loss ou repeticao do ultimo loss; None se liberado."""
    orch_cfg = orch.config.get("orchestrator", {}) if isinstance(getattr(orch, "config", None), dict) else {}
    if not bool(orch_cfg.get("cluster_block_repeat_loss_setup", True)):
        return None
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
