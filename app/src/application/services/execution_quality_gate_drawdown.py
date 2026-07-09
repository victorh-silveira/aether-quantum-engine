"""Limites de quality gate proporcionais ao drawdown de recovery."""

from __future__ import annotations

from typing import Any

from src.domain.risk.consensus_stake_penalty import resolve_session_base_unit


RECOVERY_LIGHT_MARGIN_BASE = 0.06
RECOVERY_LIGHT_MARGIN_SLOPE = 0.06


def resolve_session_stake_unit(risk_manager: Any | None, exec_cfg: dict) -> float:
    """Resolve unidade base U da sessao para amortecimento proporcional ao drawdown."""
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
        for key in ("base_stake", "kelly_base", "martingale_base"):
            base = float(kelly_cfg.get(key, 0.0) or 0.0)
            if base > 0.0:
                return max(resolve_session_base_unit(bankroll, base, None), 1e-9)
    return max(bankroll * 0.0015, 1.0) if bankroll > 0.0 else 1.0


def recovery_drawdown_quality_limits(
    recovery_limits: dict[str, float],
    regular_limits: dict[str, float],
    *,
    pending: float,
    session_unit: float,
) -> tuple[float, float]:
    """Calcula pisos elasticos de recovery proporcionais ao passivo sobre a unidade U."""
    unit = max(float(session_unit), 1e-9)
    pending_abs = max(0.0, float(pending))
    if pending_abs <= unit:
        ratio = pending_abs / unit
        margin = RECOVERY_LIGHT_MARGIN_BASE + (RECOVERY_LIGHT_MARGIN_SLOPE * ratio)
        edge = float(regular_limits["min_payoff_edge"])
        return margin, edge
    return float(recovery_limits["min_direction_margin"]), float(recovery_limits["min_payoff_edge"])
