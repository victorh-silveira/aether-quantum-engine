"""Motivos textuais e mensagens de log do quality gate."""

from __future__ import annotations


_QUALITY_GUARD_LOG_PREFIX = "[AETHER] QUALITY_GUARD |"
_EXECUTION_FLOW_LOG_PREFIX = "[AETHER] EXECUTION_FLOW |"

__all__ = [
    "build_quality_gate_reason",
    "format_quality_guard_log_message",
    "format_quality_guard_reject_message",
]


def margin_reject_clause(dir_margin: float, min_margin: float) -> str:
    """Formata clausula de rejeicao por margem direcional TCN insuficiente."""
    return f"[TCN Margin {dir_margin:.2f} < min {min_margin:.2f}]"


def edge_reject_clause(payoff_edge: float, min_edge: float) -> str:
    """Formata clausula de rejeicao por payoff meta-classificador insuficiente."""
    return f"[Meta Payoff {payoff_edge:.2f} < min {min_edge:.2f}]"


def build_quality_gate_reason(
    *,
    dir_margin: float,
    min_margin: float,
    payoff_edge: float,
    min_edge: float,
    margin_fail: bool,
    edge_fail: bool,
    meta_applied: bool,
    calibration_mode: str | None = None,
    gate_reason: str | None = None,
) -> str:
    """Monta motivo textual de rejeicao por margem TCN (payoff meta desativado)."""
    _ = (payoff_edge, min_edge, edge_fail, meta_applied)
    if str(calibration_mode or "") == "neutral_clamp" or str(gate_reason or "") == "neutral_clamp":
        return "neutral_clamp"
    if margin_fail:
        return margin_reject_clause(dir_margin, min_margin)
    return ""


def format_quality_guard_log_message(
    cycle_id: int,
    reason: str,
    *,
    linear: int,
    pending_loss: float,
) -> str:
    """Monta log estruturado de suspensao cooperativa do quality gate por ciclo."""
    cycle_label = f"C{int(cycle_id):04d}"
    return (
        f"{_QUALITY_GUARD_LOG_PREFIX} Ciclo {cycle_label} descartado. "
        f"Motivo: {reason} | linear={int(linear)} pending_loss=${float(pending_loss):.2f}"
    )


def format_quality_guard_reject_message(
    cycle_id: int,
    reason: str,
    *,
    linear: int,
    pending_loss: float,
) -> str:
    """Monta log estruturado de rejeicao do meta-regressor por ciclo."""
    cycle_label = f"C{int(cycle_id):04d}"
    return (
        f"{_EXECUTION_FLOW_LOG_PREFIX} Ciclo {cycle_label} suspenso por meta-regressor. "
        f"Motivo: {reason} | linear={int(linear)} pending_loss=${float(pending_loss):.2f}"
    )
