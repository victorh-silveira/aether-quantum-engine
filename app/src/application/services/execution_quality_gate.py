"""Quality gate de alta conviccao com janelas dinamicas por estado de risco."""

from __future__ import annotations

from typing import Any

from src.application.services.execution_quality_gate_config import resolve_quality_gate_config
from src.application.services.execution_quality_gate_drawdown import (
    apply_dynamic_recovery_relaxation,
    resolve_session_stake_unit,
)
from src.application.services.execution_quality_gate_margin import (
    direction_margin_from_probability,
    ensure_direction_margin,
    stamp_edge_without_direction,
    sync_direction_margin,
)
from src.application.services.execution_quality_gate_microstructure import apply_microstructure_starvation_veto
from src.application.services.execution_quality_gate_reason import (
    format_quality_guard_log_message,
    format_quality_guard_reject_message,
)
from src.application.services.execution_quality_gate_starvation import (
    apply_progressive_conviction_margin,
    apply_starvation_edge_decay,
    apply_starvation_margin_decay,
    starvation_decay_factor,
)
from src.application.services.execution_runtime_config import resolve_quality_gate_from_exec
from src.application.services.force_trade_mode import force_trade_every_cycle


__all__ = [
    "apply_quality_penalty_to_metrics",
    "apply_starvation_edge_decay",
    "direction_margin_from_probability",
    "ensure_direction_margin",
    "stamp_edge_without_direction",
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
    """Resolve ou aplica quality gate params."""
    qg = (
        resolve_quality_gate_config(exec_cfg)
        if isinstance(exec_cfg, dict) and isinstance(exec_cfg.get("quality_gate"), dict)
        else resolve_quality_gate_from_exec(exec_cfg if isinstance(exec_cfg, dict) else None)
    )
    return {
        "min_direction_margin": float(qg["min_direction_margin"]),
        "min_payoff_edge": float(qg["min_payoff_edge"]),
        "inverted_min_score": float(qg["inverted_min_score"]),
        "min_adx_normal": float(qg["min_adx_threshold"]),
        "min_meta_payoff_zscore": float(qg["min_meta_payoff_zscore"]),
        "mandatory_min_trade_score": float(qg["mandatory_min_trade_score"]),
    }


def _regular_quality_params(exec_cfg: dict) -> dict[str, float]:
    """Resolve ou aplica  regular quality params."""
    qg = (
        resolve_quality_gate_config(exec_cfg)
        if isinstance(exec_cfg, dict) and isinstance(exec_cfg.get("quality_gate"), dict)
        else resolve_quality_gate_from_exec(exec_cfg if isinstance(exec_cfg, dict) else None)
    )
    regular = qg["regular"]
    return {
        "min_direction_margin": float(regular["min_direction_margin"]),
        "min_payoff_edge": float(regular["min_payoff_edge"]),
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
    session_unit = resolve_session_stake_unit(risk_manager, exec_cfg)
    margin, edge, relax_intensity = apply_dynamic_recovery_relaxation(
        margin,
        edge,
        linear=session_linear,
        pending=pending,
        session_unit=session_unit,
    )
    if skipped_cycles_counter is not None:
        skipped = max(0, int(skipped_cycles_counter))
    elif orch is not None:
        skipped = max(0, int(getattr(orch, "_quality_skipped_cycles_counter", 0) or 0))
    else:
        skipped = 0
    if session_linear > 0:
        decayed_margin, decay_factor = apply_progressive_conviction_margin(
            margin,
            skipped,
            recovery_active=True,
            orch=orch,
        )
    else:
        decayed_margin, decay_factor = apply_starvation_margin_decay(margin, skipped, orch=orch)
    decayed_edge = apply_starvation_edge_decay(edge, skipped)
    return {
        **recovery_limits,
        "min_direction_margin": decayed_margin,
        "min_payoff_edge": decayed_edge,
        "quality_regime": regime,
        "session_linear": float(session_linear),
        "pending_loss_total": pending,
        "starvation_decay_factor": decay_factor,
        "skipped_cycles_counter": float(skipped),
        "recovery_relax_intensity": float(relax_intensity),
        "session_stake_unit": float(session_unit),
    }


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
    """Quality gate com veto duro por microestrutura e margem de direcao."""
    if force_trade_every_cycle(exec_cfg):
        metrics.pop("quality_guard_reject", None)
        metrics.pop("regime_skip_cycle", None)
        metrics.pop("quality_gate_reason", None)
        metrics["quality_gate_regime"] = "force_trade"
        metrics["quality_min_direction_margin"] = 0.0
        metrics["force_trade_every_cycle"] = True
        return True
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
    metrics["quality_gate_regime"] = str(limits["quality_regime"])
    metrics["quality_min_direction_margin"] = margin_floor
    metrics["quality_min_payoff_edge"] = float(limits["min_payoff_edge"])
    metrics["quality_starvation_decay_factor"] = float(limits["starvation_decay_factor"])
    metrics["quality_skipped_cycles_counter"] = float(limits["skipped_cycles_counter"])
    metrics["recovery_relax_intensity"] = float(limits.get("recovery_relax_intensity", 0.0))
    margin = ensure_direction_margin(metrics)
    starvation_reason = apply_microstructure_starvation_veto(
        metrics,
        exec_cfg=exec_cfg,
        risk_manager=risk_manager,
        orch=orch,
    )
    if starvation_reason is not None:
        metrics["quality_guard_reject"] = True
        metrics["regime_skip_cycle"] = True
        metrics["quality_gate_reason"] = starvation_reason
        return False
    if margin + 1e-12 < margin_floor:
        qg = resolve_quality_gate_config(exec_cfg)
        score_factor = float(qg.get("edge_without_direction_score_factor", 0.85))
        stamp_edge_without_direction(metrics, margin_floor=margin_floor, score_factor=score_factor)
        metrics["quality_guard_reject"] = True
        metrics["regime_skip_cycle"] = True
        metrics["quality_gate_reason"] = "direction_margin_gate"
        return False
    metrics.pop("edge_without_direction", None)
    metrics.pop("edge_without_direction_penalty", None)
    metrics["regime_skip_cycle"] = False
    metrics.pop("quality_gate_reason", None)
    metrics.pop("quality_guard_reject", None)
    return True


def apply_quality_penalty_to_metrics(
    metrics: dict,
    *,
    exec_cfg: dict | None = None,
    risk_manager: Any | None = None,
    skipped_cycles_counter: int | None = None,
    orch: Any | None = None,
    **kwargs,
) -> float:
    """Pontua rejeicao de qualidade sem mutar flags duras no candidato pronto."""
    if bool(metrics.get("execution_candidate_ready")):
        return 0.0
    probe = dict(metrics)
    passed = passes_execution_quality(
        probe,
        exec_cfg=exec_cfg,
        risk_manager=risk_manager,
        skipped_cycles_counter=skipped_cycles_counter,
        orch=orch,
        **kwargs,
    )
    for key in (
        "quality_gate_regime",
        "quality_min_direction_margin",
        "quality_min_payoff_edge",
        "quality_starvation_decay_factor",
        "quality_skipped_cycles_counter",
        "recovery_relax_intensity",
        "direction_margin",
    ):
        if key in probe:
            metrics[key] = probe[key]
    if passed:
        return 0.0
    reason = probe.get("quality_gate_reason")
    if isinstance(reason, str) and reason.strip():
        metrics["quality_penalty_reason"] = reason.strip()
    return 1.0
