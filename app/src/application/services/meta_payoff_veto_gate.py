"""Veto cruzado TCN-GBDT por expectativa de payoff e Z-Score negativo."""

from __future__ import annotations

import logging
from typing import Any

from src.application.services.execution_runtime_config import (
    resolve_meta_payoff_veto_config,
)
from src.application.services.meta_payoff_regression import CALIBRATION_NEUTRAL_DRIFT
from src.application.services.meta_payoff_shadow import (
    meta_hard_veto_allowed,
    meta_inverted_shadow_active,
    shadow_correlation,
    shadow_pair_count,
)
from src.domain.models.trade import TradeDirection
from src.domain.risk.risk_recovery_state import meta_payoff_veto_emergency_waiver
from src.domain.risk.soft_recovery_policy import (
    negative_zscore_veto_floor_for_risk,
)


logger = logging.getLogger("AETH")

META_PAYOFF_NEGATIVE_ZSCORE_VETO = "meta_payoff_negative_zscore_veto"
META_PAYOFF_SOFT_VETO = "meta_payoff_soft_zscore_veto"
EXECUTION_SIGNAL_VETO_REASONS = frozenset(
    {
        META_PAYOFF_NEGATIVE_ZSCORE_VETO,
        CALIBRATION_NEUTRAL_DRIFT,
    }
)
VETO_EDGE_EXPECTANCIES = frozenset({"NO_EDGE_NEUTRAL", "LOSS_EXPECTED"})
META_SOFT_VETO_MODE = "soft"
META_HARD_VETO_MODE = "hard"
RECOVERY_SEVERE_ZSCORE = -2.0
RECOVERY_CATASTROPHIC_ZSCORE = -2.75
RECOVERY_CATASTROPHIC_EDGE = -0.25


def _veto_cfg() -> dict[str, float]:
    """Carrega meta_payoff_veto de settings."""
    return resolve_meta_payoff_veto_config()


def _recovery_active(risk_manager: Any | None) -> bool:
    """True quando ha pendencia financeira ou perda linear ativa."""
    if risk_manager is None:
        return False
    if int(getattr(risk_manager, "consecutive_losses_linear", 0) or 0) > 0:
        return True
    pending_fn = getattr(risk_manager, "pending_loss_total", None)
    if callable(pending_fn):
        pending = float(pending_fn())
        if pending > 0.0:
            return True
    pending_map = getattr(risk_manager, "pending_loss", None)
    if isinstance(pending_map, dict):
        return sum(float(v) for v in pending_map.values()) > 0.0
    return False


def classify_payoff_edge_expectancy(
    predicted_edge: float,
    *,
    z_score: float | None = None,
    veto_floor: float | None = None,
) -> str:
    """Classifica expectativa tabular do meta-regressor a partir do edge e Z-Score."""
    cfg = _veto_cfg()
    floor = float(cfg["negative_zscore_threshold"]) if veto_floor is None else float(veto_floor)
    edge = float(predicted_edge)
    if edge <= 0.0:
        return "LOSS_EXPECTED"
    if z_score is not None and float(z_score) < floor:
        return "NO_EDGE_NEUTRAL"
    if edge < float(cfg["neutral_edge_floor"]):
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


def resolve_payoff_edge_expectancy(
    metrics: dict[str, Any],
    *,
    veto_floor: float | None = None,
) -> str:
    """Resolve expectativa; Z negativo sobrescreve WIN_EXPECTED explicito do meta."""
    floor = float(_veto_cfg()["negative_zscore_threshold"]) if veto_floor is None else float(veto_floor)
    z_score = meta_payoff_zscore(metrics) if meta_payoff_zscore_present(metrics) else None
    edge_raw = metrics.get("predicted_payoff_edge")
    explicit = metrics.get("edge_expectancy")
    if isinstance(explicit, str) and explicit.strip():
        expectancy = explicit.strip().upper()
        if expectancy == "WIN_EXPECTED" and z_score is not None and float(z_score) < floor:
            if edge_raw is not None and float(edge_raw) <= 0.0:
                return "LOSS_EXPECTED"
            return "NO_EDGE_NEUTRAL"
        return expectancy
    if edge_raw is None:
        return "WIN_EXPECTED"
    return classify_payoff_edge_expectancy(float(edge_raw), z_score=z_score, veto_floor=floor)


def stamp_payoff_edge_expectancy(
    metrics: dict[str, Any],
    *,
    veto_floor: float | None = None,
) -> str:
    """Garante edge_expectancy materializado nas metricas do candidato."""
    expectancy = resolve_payoff_edge_expectancy(metrics, veto_floor=veto_floor)
    metrics["edge_expectancy"] = expectancy
    return expectancy


def _apply_soft_veto(metrics: dict[str, Any]) -> None:
    """Comprime trade_score/conviction e marca soft veto de payoff."""
    cfg = _veto_cfg()
    metrics["meta_soft_veto_mode"] = META_SOFT_VETO_MODE
    metrics["meta_soft_veto_reason"] = META_PAYOFF_SOFT_VETO
    metrics["meta_veto_mode"] = META_SOFT_VETO_MODE
    base = metrics.get("trade_score")
    if base is None:
        base = metrics.get("resolved_conviction")
    if base is None:
        base = metrics.get("raw_prob", 0.5)
    compressed = max(float(cfg["soft_veto_min_score"]), float(base) * float(cfg["soft_veto_score_factor"]))
    metrics["trade_score"] = compressed
    metrics["conviction"] = compressed
    metrics["meta_soft_veto_penalty"] = max(0.0, float(base) - compressed)
    metrics["signal_status"] = "SOFT_VETO"


def should_veto_meta_payoff_negative_zscore(
    metrics: dict[str, Any],
    *,
    direction: TradeDirection,
    risk_manager: Any | None = None,
    orch: Any | None = None,
    recovery_active: bool = False,
) -> bool:
    """Soft veto por Z negativo; HARD com shadow positivo ou Z invertido anti-PnL."""
    _ = direction
    veto_floor = negative_zscore_veto_floor_for_risk(risk_manager)
    metrics["meta_payoff_veto_zscore_floor"] = float(veto_floor)
    expectancy = stamp_payoff_edge_expectancy(metrics, veto_floor=veto_floor)
    z_present = meta_payoff_zscore_present(metrics)
    z_score = meta_payoff_zscore(metrics) if z_present else None
    soft_hit = bool(
        z_present
        and z_score is not None
        and float(z_score) < float(veto_floor)
        and expectancy in VETO_EDGE_EXPECTANCIES
    )
    samples = int(metrics.get("edge_zscore_samples") or 0)
    if soft_hit and samples < 2:
        soft_hit = False
        metrics["meta_soft_veto_deferred"] = True
    severe_z = bool(
        z_present
        and z_score is not None
        and float(z_score) < float(veto_floor)
        and samples >= 2
        and expectancy in VETO_EDGE_EXPECTANCIES
    )
    if severe_z:
        soft_hit = True
        metrics.pop("meta_soft_veto_deferred", None)
    corr = shadow_correlation(orch)
    n_shadow = shadow_pair_count()
    if orch is not None:
        n_shadow = max(n_shadow, int(getattr(orch, "_meta_payoff_shadow_n", 0) or 0))
    metrics["meta_shadow_corr"] = corr
    metrics["meta_shadow_n"] = int(n_shadow)
    inverted = meta_inverted_shadow_active(orch)
    metrics["meta_shadow_inverted"] = bool(inverted)
    inverted_hit = bool(inverted and z_present and z_score is not None and float(z_score) > abs(float(veto_floor)))
    hard_allowed = meta_hard_veto_allowed(orch)
    metrics["meta_payoff_soft_veto"] = soft_hit or inverted_hit
    if not soft_hit and not inverted_hit:
        metrics["meta_veto_mode"] = "none"
        metrics["meta_soft_veto_penalty"] = 0.0
        logger.debug(
            "META_VETO_MODE=%s | shadow_corr=%s | n=%d",
            "none",
            f"{corr:+.3f}" if corr is not None else "na",
            n_shadow,
        )
        return False
    metrics["meta_veto_mode"] = META_SOFT_VETO_MODE
    waived = False
    if risk_manager is not None:
        try:
            waived = bool(
                meta_payoff_veto_emergency_waiver(
                    metrics,
                    direction=direction.name,
                    risk_manager=risk_manager,
                )
            )
        except Exception:
            waived = False
    in_recovery = bool(recovery_active) or _recovery_active(risk_manager)
    edge_raw = metrics.get("predicted_payoff_edge")
    edge_positive = edge_raw is not None and float(edge_raw) > 0.0
    inverted_recovery_soft = bool(inverted_hit and in_recovery and edge_positive)
    if inverted_recovery_soft:
        metrics["meta_shadow_inverted_recovery_soft"] = True
        inverted_hit = False
    recovery_severe = bool(
        soft_hit and z_score is not None and float(z_score) <= float(RECOVERY_SEVERE_ZSCORE) and in_recovery
    )
    recovery_catastrophic = bool(
        recovery_severe
        and z_score is not None
        and float(z_score) <= float(RECOVERY_CATASTROPHIC_ZSCORE)
        and edge_raw is not None
        and float(edge_raw) <= float(RECOVERY_CATASTROPHIC_EDGE)
    )
    recovery_severe_hard = bool(recovery_catastrophic and not edge_positive)
    metrics["meta_recovery_severe_z"] = bool(recovery_severe)
    metrics["meta_recovery_catastrophic_z"] = bool(recovery_catastrophic)
    metrics["meta_recovery_active"] = bool(in_recovery)
    if (hard_allowed or inverted_hit or recovery_severe_hard) and not waived:
        apply_meta_payoff_negative_zscore_veto(metrics)
        metrics["meta_veto_mode"] = META_HARD_VETO_MODE
        metrics["meta_soft_veto_penalty"] = 0.0
        if inverted_hit:
            metrics["gate_reason"] = "meta_shadow_inverted_veto"
        if recovery_severe_hard:
            metrics["meta_recovery_severe_z_veto"] = True
        logger.info(
            "META_HARD_VETO | META_VETO_MODE=%s | shadow_corr=%+.3f | n=%d | z=%.3f | inverted=%s | recovery=%s",
            META_HARD_VETO_MODE,
            float(corr or 0.0),
            n_shadow,
            float(z_score or 0.0),
            str(bool(inverted)).lower(),
            str(bool(in_recovery)).lower(),
        )
        return True
    _apply_soft_veto(metrics)
    if recovery_severe:
        metrics["meta_recovery_severe_z_soft"] = True
    logger.info(
        "META_VETO_MODE=%s | shadow_corr=%s | n=%d | z=%.3f",
        META_SOFT_VETO_MODE,
        f"{corr:+.3f}" if corr is not None else "na",
        n_shadow,
        float(z_score or 0.0),
    )
    return False


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
