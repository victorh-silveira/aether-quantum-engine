"""Helpers de margem direcional e estado de risco (sem quality veto)."""

from __future__ import annotations

from typing import Any

from src.application.services.execution_quality_gate_margin import (
    direction_margin_from_probability,
    ensure_direction_margin,
    stamp_edge_without_direction,
    sync_direction_margin,
)


__all__ = [
    "apply_quality_penalty_to_metrics",
    "direction_margin_from_probability",
    "ensure_direction_margin",
    "stamp_edge_without_direction",
    "sync_direction_margin",
    "read_risk_session_state",
]


def read_risk_session_state(
    risk_manager: Any | None,
    *,
    linear: int | None = None,
    pending_loss_total: float | None = None,
) -> tuple[int, float]:
    """Extrai perdas lineares e passivo pendente do RiskManager ou overrides explicitos."""
    if linear is not None:
        session_linear = int(linear)
    elif risk_manager is not None:
        session_linear = int(
            getattr(risk_manager, "consecutive_losses_linear", getattr(risk_manager, "linear", 0)) or 0,
        )
    else:
        session_linear = 0
    if pending_loss_total is not None:
        pending = float(pending_loss_total)
    elif risk_manager is not None:
        total_fn = getattr(risk_manager, "pending_loss_total", None)
        if callable(total_fn):
            pending = float(total_fn())
        else:
            pending_map = getattr(risk_manager, "pending_loss", None)
            pending = float(sum(pending_map.values())) if isinstance(pending_map, dict) else 0.0
    else:
        pending = 0.0
    return session_linear, pending


def apply_quality_penalty_to_metrics(
    metrics: dict,
    *,
    exec_cfg: dict | None = None,
    risk_manager: Any | None = None,
    skipped_cycles_counter: int | None = None,
    orch: Any | None = None,
    **kwargs,
) -> float:
    """Telemetria de margem sem rejeicao de qualidade."""
    _ = (exec_cfg, risk_manager, skipped_cycles_counter, orch, kwargs)
    ensure_direction_margin(metrics)
    return 0.0
