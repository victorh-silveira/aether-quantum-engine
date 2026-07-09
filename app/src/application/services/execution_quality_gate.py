"""Quality gate de alta conviccao com janelas dinamicas por estado de risco."""

from __future__ import annotations

from typing import Any

from src.application.services.execution_quality_gate_reason import (
    build_quality_gate_reason,
    format_quality_guard_log_message,
    format_quality_guard_reject_message,
)
from src.application.services.execution_quality_gate_starvation import (
    apply_starvation_margin_decay,
    starvation_decay_factor,
)


MANDATORY_MIN_TRADE_SCORE_DEFAULT = 0.72
MIN_DIRECTION_MARGIN_DEFAULT = 0.11
MIN_PAYOFF_EDGE_DEFAULT = 0.04
REGULAR_MIN_DIRECTION_MARGIN_DEFAULT = 0.06
REGULAR_MIN_PAYOFF_EDGE_DEFAULT = 0.01

__all__ = [
    "apply_quality_penalty_to_metrics",
    "direction_margin_from_probability",
    "ensure_direction_margin",
    "sync_direction_margin",
    "format_quality_guard_log_message",
    "format_quality_guard_reject_message",
    "passes_execution_quality",
    "quality_gate_params",
    "read_risk_session_state",
    "resolve_dynamic_quality_limits",
    "starvation_decay_factor",
]


def quality_gate_params(exec_cfg: dict) -> dict[str, float]:
    """Le limites de recovery configurados em orchestrator.execution.quality_gate."""
    chunk = exec_cfg.get("quality_gate") if isinstance(exec_cfg, dict) else {}
    if not isinstance(chunk, dict):
        chunk = {}
    return {
        "min_direction_margin": float(chunk.get("min_direction_margin", MIN_DIRECTION_MARGIN_DEFAULT)),
        "min_payoff_edge": float(chunk.get("min_payoff_edge", MIN_PAYOFF_EDGE_DEFAULT)),
        "inverted_min_score": float(chunk.get("inverted_min_score", 0.0)),
        "min_adx_normal": float(chunk.get("min_adx_normal", 0.0)),
        "min_meta_payoff_zscore": float(chunk.get("min_meta_payoff_zscore", 0.5)),
    }


def _regular_quality_params(exec_cfg: dict) -> dict[str, float]:
    """Le limites elasticos do regime regular em orchestrator.execution.quality_gate."""
    chunk = exec_cfg.get("quality_gate") if isinstance(exec_cfg, dict) else {}
    regular = chunk.get("regular") if isinstance(chunk, dict) else {}
    if not isinstance(regular, dict):
        regular = {}
    return {
        "min_direction_margin": float(
            regular.get("min_direction_margin", REGULAR_MIN_DIRECTION_MARGIN_DEFAULT),
        ),
        "min_payoff_edge": float(regular.get("min_payoff_edge", REGULAR_MIN_PAYOFF_EDGE_DEFAULT)),
    }


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
            raw_pending = getattr(risk_manager, "pending_loss", {})
            pending = sum(float(value) for value in raw_pending.values()) if isinstance(raw_pending, dict) else 0.0
    else:
        pending = 0.0
    return session_linear, pending


def resolve_dynamic_quality_limits(
    exec_cfg: dict,
    *,
    risk_manager: Any | None = None,
    linear: int | None = None,
    pending_loss_total: float | None = None,
    override_margin: float | None = None,
    override_edge: float | None = None,
    skipped_cycles_counter: int | None = None,
    orch: Any | None = None,
) -> dict[str, float | str]:
    """Calibra limites elasticos conforme regime regular ou recovery da sessao."""
    recovery_limits = quality_gate_params(exec_cfg)
    regular_limits = _regular_quality_params(exec_cfg)
    session_linear, pending = read_risk_session_state(
        risk_manager,
        linear=linear,
        pending_loss_total=pending_loss_total,
    )
    recovery_active = session_linear > 0 or pending > 0.0
    if recovery_active:
        selected = recovery_limits
        regime = "recovery"
    else:
        selected = regular_limits
        regime = "regular"
    margin = float(selected["min_direction_margin"])
    edge = float(selected["min_payoff_edge"])
    if override_margin is not None:
        margin = float(override_margin)
    if override_edge is not None:
        edge = float(override_edge)
    if skipped_cycles_counter is not None:
        skipped = max(0, int(skipped_cycles_counter))
    elif orch is not None:
        skipped = max(0, int(getattr(orch, "_quality_skipped_cycles_counter", 0) or 0))
    else:
        skipped = 0
    decayed_margin, decay_factor = apply_starvation_margin_decay(margin, skipped, orch=orch)
    return {
        **recovery_limits,
        "min_direction_margin": decayed_margin,
        "min_payoff_edge": edge,
        "quality_regime": regime,
        "session_linear": float(session_linear),
        "pending_loss_total": pending,
        "starvation_decay_factor": decay_factor,
        "skipped_cycles_counter": float(skipped),
    }


def direction_margin_from_probability(call_probability: float, *, direction: str | None = None) -> float:
    """Calcula distancia da confianca lateral escolhida ao centro neutro 0.50."""
    call_score = max(0.0, min(1.0, float(call_probability)))
    side_probability = call_score
    if str(direction or "").upper() == "PUT":
        side_probability = 1.0 - call_score
    return abs(side_probability - 0.5)


def ensure_direction_margin(metrics: dict) -> float:
    """Garante direction_margin a partir da probabilidade calibrada ou bruta."""
    prob = metrics.get("calibrated_prob", metrics.get("raw_prob"))
    if prob is not None:
        direction = metrics.get("exec_direction") or metrics.get("resolved_direction") or metrics.get("dl_direction")
        margin = direction_margin_from_probability(
            float(prob),
            direction=str(direction) if direction is not None else None,
        )
    else:
        stored = metrics.get("direction_margin")
        margin = float(stored) if stored is not None else 0.0
    metrics["direction_margin"] = margin
    return margin


def sync_direction_margin(metrics: dict, *, direction: str) -> float:
    """Atualiza direction_margin a partir da probabilidade calibrada ou scores laterais."""
    prob = metrics.get("calibrated_prob", metrics.get("raw_prob"))
    if prob is not None:
        margin = direction_margin_from_probability(float(prob), direction=direction)
    else:
        margin = abs(float(metrics["direction_call_score"]) - float(metrics["direction_put_score"]))
    metrics["direction_margin"] = margin
    return margin


def passes_execution_quality(
    metrics: dict,
    *,
    exec_cfg: dict | None = None,
    risk_manager: Any | None = None,
    linear: int | None = None,
    pending_loss_total: float | None = None,
    min_direction_margin: float | None = None,
    min_payoff_edge: float | None = None,
    skipped_cycles_counter: int | None = None,
    orch: Any | None = None,
    **_kwargs,
) -> bool:
    """Valida margem direcional e payoff previsto com janelas dinamicas de qualidade."""
    limits = resolve_dynamic_quality_limits(
        exec_cfg or {},
        risk_manager=risk_manager,
        linear=linear,
        pending_loss_total=pending_loss_total,
        override_margin=min_direction_margin,
        override_edge=min_payoff_edge,
        skipped_cycles_counter=skipped_cycles_counter,
        orch=orch,
    )
    margin_floor = float(limits["min_direction_margin"])
    edge_floor = float(limits["min_payoff_edge"])
    metrics["quality_gate_regime"] = str(limits["quality_regime"])
    metrics["quality_min_direction_margin"] = margin_floor
    metrics["quality_min_payoff_edge"] = edge_floor
    metrics["quality_starvation_decay_factor"] = float(limits["starvation_decay_factor"])
    metrics["quality_skipped_cycles_counter"] = float(limits["skipped_cycles_counter"])
    margin = ensure_direction_margin(metrics)
    margin_fail = margin <= margin_floor + 1e-12
    meta_applied = bool(metrics.get("meta_classifier_applied") or metrics.get("meta_applied"))
    payoff_edge = float(metrics.get("predicted_payoff_edge", 0.0))
    edge_fail = meta_applied and payoff_edge + 1e-12 < edge_floor
    if margin_fail or edge_fail:
        metrics["regime_skip_cycle"] = True
        metrics["quality_gate_reason"] = build_quality_gate_reason(
            dir_margin=margin,
            min_margin=margin_floor,
            payoff_edge=payoff_edge,
            min_edge=edge_floor,
            margin_fail=margin_fail,
            edge_fail=edge_fail,
            meta_applied=meta_applied,
        )
        return False
    metrics["regime_skip_cycle"] = False
    metrics.pop("quality_gate_reason", None)
    return True


def apply_quality_penalty_to_metrics(
    metrics: dict,
    *,
    exec_cfg: dict | None = None,
    risk_manager: Any | None = None,
    **_kwargs,
) -> float:
    """Aplica veto de qualidade retornando penalidade unitaria quando o gate reprova."""
    if passes_execution_quality(metrics, exec_cfg=exec_cfg, risk_manager=risk_manager):
        return 0.0
    return 1.0
