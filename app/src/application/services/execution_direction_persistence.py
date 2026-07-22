"""Guard de persistencia de lado com flip toxico controlado."""

from __future__ import annotations

from src.application.services.direction_loss_tracker import consecutive_direction_losses
from src.application.services.direction_persistence_guard import evaluate_direction_persistence_guard
from src.application.services.execution_direction_checks import sync_entry_metrics
from src.application.services.execution_runtime_config import resolve_direction_persistence_config
from src.domain.models.trade import TradeDirection


def _apply_persistence_guard_skip(
    entry: dict,
    metrics: dict,
    dl_dir: TradeDirection,
    *,
    symbol: str | None,
    peer_entry: dict | None,
    cycle_id: int,
    infra_cfg: dict | None,
    force: bool = False,
) -> TradeDirection | None:
    """None = bloqueio total; TradeDirection = lado permitido (pode ser o oposto)."""
    if force:
        metrics.pop("persistence_guard_skip", None)
        metrics.pop("quality_guard_reject", None)
        return dl_dir
    guarded = evaluate_direction_persistence_guard(
        symbol, dl_dir, dl_dir, metrics, entry=entry, peer_entry=peer_entry, cycle_id=cycle_id, infra_cfg=infra_cfg
    )
    if guarded is not None:
        return guarded
    opposite = TradeDirection.CALL if dl_dir == TradeDirection.PUT else TradeDirection.PUT
    threshold = int(resolve_direction_persistence_config()["same_direction_count_threshold"])
    if symbol and consecutive_direction_losses(symbol, opposite.name) < threshold:
        metrics.pop("persistence_guard_skip", None)
        metrics.pop("quality_guard_reject", None)
        metrics.pop("gate_reason", None)
        metrics["persistence_guard_flip"] = opposite.name
        metrics["side_eq_toxic_escape"] = True
        metrics["dl_direction"] = opposite.name
        metrics["exec_direction"] = opposite.name
        metrics["resolved_direction"] = opposite.name
        sync_entry_metrics(entry, metrics)
        return opposite
    metrics["gate_reason"] = str(metrics.get("gate_reason") or "persistence_guard_skip")
    metrics["persistence_guard_skip"] = True
    metrics["quality_guard_reject"] = True
    sync_entry_metrics(entry, metrics)
    return None
