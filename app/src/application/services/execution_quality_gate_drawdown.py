"""Dynamic Recovery Relaxation: pisos elasticos de quality gate sob passivo."""

from __future__ import annotations

from typing import Any

from src.application.services.execution_runtime_config import resolve_quality_gate_from_exec
from src.domain.risk.consensus_stake_penalty import resolve_session_base_unit


def _qg(exec_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve ou aplica  qg."""
    return resolve_quality_gate_from_exec(exec_cfg)


def resolve_session_stake_unit(risk_manager: Any | None, exec_cfg: dict) -> float:
    """Resolve unidade de stake da sessao para relaxation de recovery."""
    chunk = exec_cfg.get("quality_gate") if isinstance(exec_cfg, dict) else {}
    if isinstance(chunk, dict):
        configured = chunk.get("session_base_unit")
        if configured is not None:
            return max(float(configured), 1e-9)
    if risk_manager is None:
        return 1.0
    for attr in ("dlambert_unit", "session_base_unit"):
        value = float(getattr(risk_manager, attr, 0.0) or 0.0)
        if value > 0.0:
            return value
    bankroll = float(getattr(risk_manager, "initial_bankroll", 0.0) or 0.0)
    kelly_cfg = getattr(risk_manager, "kelly_config", None)
    if isinstance(kelly_cfg, dict):
        for key in ("base_stake", "kelly_base"):
            base = float(kelly_cfg.get(key, 0.0) or 0.0)
            if base > 0.0:
                return max(resolve_session_base_unit(bankroll, base, None), 1e-9)
    qg = _qg(exec_cfg if isinstance(exec_cfg, dict) else None)
    pct = float(qg["recovery_relax"]["session_stake_unit_bankroll_pct"])
    return max(bankroll * pct, 1.0) if bankroll > 0.0 else 1.0


def apply_dynamic_recovery_relaxation(
    margin: float,
    edge: float,
    *,
    linear: int,
    pending: float,
    session_unit: float,
    exec_cfg: dict[str, Any] | None = None,
) -> tuple[float, float, float]:
    """Resolve ou aplica apply dynamic recovery relaxation."""
    relax = _qg(exec_cfg)["recovery_relax"]
    if int(linear) < int(relax["min_linear"]) or float(pending) <= 0.0:
        return float(margin), float(edge), 0.0
    unit = max(float(session_unit), 1e-9)
    intensity = min(1.0, max(0.0, float(pending)) / (float(relax["full_pending_units"]) * unit))
    relaxed_margin = float(margin) - (float(margin) - float(relax["margin_floor"])) * intensity
    relaxed_edge = float(edge) - (float(edge) - float(relax["edge_floor"])) * intensity
    return relaxed_margin, relaxed_edge, float(intensity)


def recovery_neutral_edge_zscore_waiver(
    edge: float,
    z_edge: float,
    *,
    linear: int,
    pending: float,
    exec_cfg: dict[str, Any] | None = None,
) -> bool:
    """Resolve ou aplica recovery neutral edge zscore waiver."""
    qg = _qg(exec_cfg)
    relax = qg["recovery_relax"]
    neutral = qg["neutral_meta_payoff"]
    if int(linear) < int(relax["min_linear"]) or float(pending) <= 0.0:
        return False
    if float(edge) + 1e-12 < float(neutral["lo"]):
        return False
    if float(edge) - 1e-12 > float(neutral["hi"]):
        return False
    return float(z_edge) > float(relax["edge_zscore_waiver"]) + 1e-12


def recovery_drawdown_quality_limits(
    recovery_limits: dict[str, float],
    regular_limits: dict[str, float],
    *,
    pending: float,
    session_unit: float,
    linear: int | None = None,
    exec_cfg: dict[str, Any] | None = None,
) -> tuple[float, float]:
    """Resolve ou aplica recovery drawdown quality limits."""
    _ = regular_limits
    relax = _qg(exec_cfg)["recovery_relax"]
    linear_val = int(relax["min_linear"]) if linear is None else int(linear)
    margin, edge, _intensity = apply_dynamic_recovery_relaxation(
        float(recovery_limits["min_direction_margin"]),
        float(recovery_limits["min_payoff_edge"]),
        linear=linear_val,
        pending=float(pending),
        session_unit=float(session_unit),
        exec_cfg=exec_cfg,
    )
    return margin, edge
