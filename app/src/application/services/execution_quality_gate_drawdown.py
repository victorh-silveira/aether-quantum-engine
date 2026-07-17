"""Dynamic Recovery Relaxation: pisos elasticos de quality gate sob passivo."""

from __future__ import annotations

from typing import Any

from src.domain.risk.consensus_stake_penalty import resolve_session_base_unit


RECOVERY_RELAX_MIN_LINEAR = 2
RECOVERY_RELAX_MARGIN_FLOOR = 0.01
RECOVERY_RELAX_EDGE_FLOOR = -0.55
RECOVERY_RELAX_FULL_PENDING_UNITS = 8.0
NEUTRAL_META_PAYOFF_LO = -0.05
NEUTRAL_META_PAYOFF_HI = 0.04
RECOVERY_EDGE_ZSCORE_WAIVER = 0.5


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


def apply_dynamic_recovery_relaxation(
    margin: float,
    edge: float,
    *,
    linear: int,
    pending: float,
    session_unit: float,
) -> tuple[float, float, float]:
    """Reduz linearmente pisos de TCN Margin e Meta Payoff conforme o passivo."""
    if int(linear) < RECOVERY_RELAX_MIN_LINEAR or float(pending) <= 0.0:
        return float(margin), float(edge), 0.0
    unit = max(float(session_unit), 1e-9)
    intensity = min(1.0, max(0.0, float(pending)) / (RECOVERY_RELAX_FULL_PENDING_UNITS * unit))
    relaxed_margin = float(margin) - (float(margin) - RECOVERY_RELAX_MARGIN_FLOOR) * intensity
    relaxed_edge = float(edge) - (float(edge) - RECOVERY_RELAX_EDGE_FLOOR) * intensity
    return relaxed_margin, relaxed_edge, float(intensity)


def recovery_neutral_edge_zscore_waiver(
    edge: float,
    z_edge: float,
    *,
    linear: int,
    pending: float,
) -> bool:
    """True quando edge neutro com Z>0.5 libera trade de recovery (vacuo GBDT)."""
    if int(linear) < RECOVERY_RELAX_MIN_LINEAR or float(pending) <= 0.0:
        return False
    if float(edge) + 1e-12 < NEUTRAL_META_PAYOFF_LO:
        return False
    if float(edge) - 1e-12 > NEUTRAL_META_PAYOFF_HI:
        return False
    return float(z_edge) > RECOVERY_EDGE_ZSCORE_WAIVER + 1e-12


def recovery_drawdown_quality_limits(
    recovery_limits: dict[str, float],
    regular_limits: dict[str, float],
    *,
    pending: float,
    session_unit: float,
    linear: int = RECOVERY_RELAX_MIN_LINEAR,
) -> tuple[float, float]:
    """Calcula pisos elasticos de recovery proporcionais ao passivo sobre a unidade U."""
    _ = regular_limits
    margin, edge, _intensity = apply_dynamic_recovery_relaxation(
        float(recovery_limits["min_direction_margin"]),
        float(recovery_limits["min_payoff_edge"]),
        linear=int(linear),
        pending=float(pending),
        session_unit=float(session_unit),
    )
    return margin, edge
