"""Piso dinamico de edge meta e marcacao de ciclo resolvido."""

from __future__ import annotations

from typing import Any

from src.application.services.execution_quality_gate import resolve_dynamic_quality_limits
from src.application.services.execution_quality_gate_config import resolve_quality_gate_config


def _resolve_meta_edge_floor(
    metrics: dict,
    *,
    exec_cfg: dict | None,
    skipped_cycles_counter: int | None,
    orch: Any | None,
    recovery_active: bool = False,
    risk_manager: Any | None = None,
) -> float:
    """Resolve piso dinamico de edge meta com starvation e recovery."""
    skipped = 0
    if skipped_cycles_counter is not None:
        skipped = max(0, int(skipped_cycles_counter))
    elif orch is not None:
        skipped = max(0, int(getattr(orch, "_quality_skipped_cycles_counter", 0) or 0))
    cfg = exec_cfg if isinstance(exec_cfg, dict) else {}
    limits = resolve_dynamic_quality_limits(
        cfg,
        risk_manager=risk_manager,
        skipped_cycles_counter=skipped,
        orch=orch,
    )
    floor = float(limits["min_payoff_edge"])
    metrics["quality_skipped_cycles_counter"] = float(limits.get("skipped_cycles_counter", skipped))
    in_recovery = bool(recovery_active)
    if not in_recovery and risk_manager is not None:
        raw_linear = getattr(risk_manager, "consecutive_losses_linear", 0)
        linear = int(raw_linear) if isinstance(raw_linear, int | float) and not isinstance(raw_linear, bool) else 0
        pending_map = getattr(risk_manager, "pending_loss", None)
        pending = float(sum(pending_map.values())) if isinstance(pending_map, dict) else 0.0
        in_recovery = linear > 0 or pending > 0.0
    qg = resolve_quality_gate_config(cfg if cfg else None)
    if in_recovery:
        relax_floor = float(qg["recovery_relax"]["edge_floor"])
        if floor > relax_floor:
            floor = relax_floor
            metrics["meta_negative_edge_recovery_waiver"] = True
    metrics["quality_min_payoff_edge"] = floor
    return floor


def _negative_edge_skip(
    metrics: dict,
    predicted_edge: float,
    *,
    force: bool,
    meta_applied: bool,
    exec_cfg: dict | None = None,
    skipped_cycles_counter: int | None = None,
    orch: Any | None = None,
    recovery_active: bool = False,
    risk_manager: Any | None = None,
) -> bool:
    """Bloqueia trades com edge abaixo do piso (min_payoff_edge via quality_gate)."""
    if force or not meta_applied or float(metrics.get("senior_trader_conviction", 0.0) or 0.0) >= 0.56:
        return False
    edge = metrics.get("predicted_payoff_edge")
    edge_v = float(predicted_edge) if edge is None else float(edge)
    floor = _resolve_meta_edge_floor(
        metrics,
        exec_cfg=exec_cfg,
        skipped_cycles_counter=skipped_cycles_counter,
        orch=orch,
        recovery_active=recovery_active,
        risk_manager=risk_manager,
    )
    if edge_v + 1e-12 >= floor:
        if edge_v <= 0.0:
            metrics["meta_negative_edge_starvation_waiver"] = True
            metrics["meta_edge_floor"] = floor
        return False
    metrics["meta_edge_floor"] = floor
    return True


def _stamp_direction_resolved_cycle(entry: dict, cycle_id: int) -> None:
    """Marca o ciclo atual como ja resolvido nas metricas do candidato."""
    if int(cycle_id or 0) <= 0:
        return
    bag = entry.get("metrics")
    if not isinstance(bag, dict):
        entry["metrics"] = {"_direction_resolved_cycle": int(cycle_id)}
        return
    bag["_direction_resolved_cycle"] = int(cycle_id)
