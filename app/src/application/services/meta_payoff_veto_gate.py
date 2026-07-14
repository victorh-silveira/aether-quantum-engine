"""Veto cruzado TCN-GBDT por expectativa de payoff e Z-Score negativo."""

from __future__ import annotations

from typing import Any

from src.application.services.meta_payoff_regression import CALIBRATION_NEUTRAL_DRIFT
from src.domain.models.trade import TradeDirection
from src.domain.risk.risk_recovery_state import meta_payoff_veto_emergency_waiver


META_PAYOFF_NEGATIVE_ZSCORE_VETO = "meta_payoff_negative_zscore_veto"
EXECUTION_SIGNAL_VETO_REASONS = frozenset(
    {
        META_PAYOFF_NEGATIVE_ZSCORE_VETO,
        CALIBRATION_NEUTRAL_DRIFT,
    }
)
NEGATIVE_ZSCORE_VETO_THRESHOLD = -0.20
VETO_EDGE_EXPECTANCIES = frozenset({"NO_EDGE_NEUTRAL", "LOSS_EXPECTED"})
NEUTRAL_EDGE_FLOOR = 0.04


def classify_payoff_edge_expectancy(predicted_edge: float, *, z_score: float | None = None) -> str:
    """Classifica expectativa tabular do meta-regressor a partir do edge e Z-Score."""
    edge = float(predicted_edge)
    if edge <= 0.0:
        return "LOSS_EXPECTED"
    if z_score is not None and float(z_score) < NEGATIVE_ZSCORE_VETO_THRESHOLD:
        return "NO_EDGE_NEUTRAL"
    if edge < NEUTRAL_EDGE_FLOOR:
        return "NO_EDGE_NEUTRAL"
    return "WIN_EXPECTED"


def meta_payoff_zscore_present(metrics: dict[str, Any]) -> bool:
    """True quando o buffer movel ja anexou Z-Score de payoff nas metricas."""
    return metrics.get("meta_payoff_edge_zscore") is not None or metrics.get("edge_zscore") is not None


def meta_payoff_zscore(metrics: dict[str, Any]) -> float:
    """Le Z-Score de payoff anexado pelo buffer movel Redis."""
    for key in ("meta_payoff_edge_zscore", "edge_zscore"):
        raw = metrics.get(key)
        if raw is not None:
            return float(raw)
    return 0.0


def resolve_payoff_edge_expectancy(metrics: dict[str, Any]) -> str:
    """Resolve expectativa; Z negativo sobrescreve WIN_EXPECTED explicito do meta."""
    z_score = meta_payoff_zscore(metrics) if meta_payoff_zscore_present(metrics) else None
    edge_raw = metrics.get("predicted_payoff_edge")
    explicit = metrics.get("edge_expectancy")
    if isinstance(explicit, str) and explicit.strip():
        expectancy = explicit.strip().upper()
        if expectancy == "WIN_EXPECTED" and z_score is not None and float(z_score) < NEGATIVE_ZSCORE_VETO_THRESHOLD:
            if edge_raw is not None and float(edge_raw) <= 0.0:
                return "LOSS_EXPECTED"
            return "NO_EDGE_NEUTRAL"
        return expectancy
    if edge_raw is None:
        return "WIN_EXPECTED"
    return classify_payoff_edge_expectancy(float(edge_raw), z_score=z_score)


def stamp_payoff_edge_expectancy(metrics: dict[str, Any]) -> str:
    """Garante edge_expectancy materializado nas metricas do candidato."""
    expectancy = resolve_payoff_edge_expectancy(metrics)
    metrics["edge_expectancy"] = expectancy
    return expectancy


def should_veto_meta_payoff_negative_zscore(
    metrics: dict[str, Any],
    *,
    direction: TradeDirection,
    risk_manager: Any | None = None,
) -> bool:
    """True quando expectativa neutra/perda combina com Z-Score abaixo do piso."""
    expectancy = stamp_payoff_edge_expectancy(metrics)
    if expectancy not in VETO_EDGE_EXPECTANCIES:
        return False
    z_score = meta_payoff_zscore(metrics)
    if z_score >= NEGATIVE_ZSCORE_VETO_THRESHOLD:
        return False
    if meta_payoff_veto_emergency_waiver(
        metrics,
        direction=direction.name,
        risk_manager=risk_manager,
    ):
        metrics["meta_payoff_veto_waived"] = True
        return False
    return True


def is_execution_signal_vetoed(metrics: dict[str, Any] | None) -> bool:
    """True quando gate_reason indica veto absoluto de direcao."""
    if not isinstance(metrics, dict):
        return False
    return str(metrics.get("gate_reason") or "") in EXECUTION_SIGNAL_VETO_REASONS


def apply_meta_payoff_negative_zscore_veto(metrics: dict[str, Any]) -> None:
    """Invalida direcao e score para SKIP absoluto do ativo."""
    metrics["resolved_direction"] = None
    metrics["exec_direction"] = None
    metrics["gate_reason"] = META_PAYOFF_NEGATIVE_ZSCORE_VETO
    metrics["trade_score"] = None
    metrics["conviction"] = None
    metrics["signal_status"] = "SKIP"
