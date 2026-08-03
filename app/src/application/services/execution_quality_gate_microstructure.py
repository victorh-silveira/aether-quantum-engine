"""Veto de inanicao de mercado por ADX, vol_ratio e val_accuracy."""

from __future__ import annotations

from typing import Any

from src.application.services.execution_quality_gate_starvation import starvation_decay_factor
from src.application.services.execution_tcn_conviction import tcn_high_conviction_active


MICROSTRUCTURE_STARVATION_REASONS = frozenset(
    {
        "adx_starvation",
        "vol_ratio_starvation",
        "val_accuracy_gate",
    }
)
DIRECTION_MARGIN_GATE_REASON = "direction_margin_gate"
HARD_QUALITY_REJECT_REASONS = MICROSTRUCTURE_STARVATION_REASONS | {DIRECTION_MARGIN_GATE_REASON}

__all__ = [
    "DIRECTION_MARGIN_GATE_REASON",
    "HARD_QUALITY_REJECT_REASONS",
    "MICROSTRUCTURE_STARVATION_REASONS",
    "apply_microstructure_starvation_veto",
    "is_hard_quality_reject_reason",
    "is_microstructure_starvation_reason",
    "resolve_min_adx_threshold",
    "resolve_min_validation_accuracy_gate",
    "resolve_min_vol_ratio",
    "resolve_skipped_cycles",
]


def is_microstructure_starvation_reason(reason: Any) -> bool:
    """True quando a razao de rejeicao e veto duro de microestrutura."""
    return str(reason or "") in MICROSTRUCTURE_STARVATION_REASONS


def is_hard_quality_reject_reason(reason: Any) -> bool:
    """True quando a razao deve abortar o candidato no resolver."""
    return str(reason or "") in HARD_QUALITY_REJECT_REASONS


def resolve_skipped_cycles(
    *,
    skipped_cycles_counter: int | None = None,
    orch: Any | None = None,
) -> int:
    """Resolve contador de ciclos pulados por quality/inanicao."""
    if skipped_cycles_counter is not None:
        return max(0, int(skipped_cycles_counter))
    if orch is not None:
        return max(0, int(getattr(orch, "_quality_skipped_cycles_counter", 0) or 0))
    return 0


def _indicator_float(metrics: dict, key: str) -> float | None:
    """Le indicador float de blocos macro/micro/indicators."""
    for block_name in ("indicators", "macro_indicators", "micro_indicators"):
        block = metrics.get(block_name)
        if not isinstance(block, dict) or block.get(key) is None:
            continue
        try:
            return float(block[key])
        except (TypeError, ValueError):
            return None
    if metrics.get(key) is None:
        return None
    try:
        return float(metrics[key])
    except (TypeError, ValueError):
        return None


def resolve_min_adx_threshold(exec_cfg: dict | None) -> float:
    """Resolve piso de ADX em quality_gate (min_adx_threshold ou min_adx_normal)."""
    chunk = exec_cfg.get("quality_gate") if isinstance(exec_cfg, dict) else {}
    if not isinstance(chunk, dict):
        return 0.0
    if chunk.get("min_adx_threshold") is not None:
        return float(chunk["min_adx_threshold"])
    if chunk.get("min_adx_normal") is not None:
        return float(chunk["min_adx_normal"])
    return 0.0


def resolve_min_vol_ratio(orch: Any | None, exec_cfg: dict | None) -> float:
    """Resolve vol_ratio_min de deep_learning.indicator_gating quando enabled."""
    config = getattr(orch, "config", None) if orch is not None else None
    if not isinstance(config, dict):
        risk = getattr(orch, "risk_manager", None) if orch is not None else None
        config = getattr(risk, "config", None) if risk is not None else None
    if not isinstance(config, dict):
        config = {}
    dl = config.get("deep_learning")
    gating = dl.get("indicator_gating") if isinstance(dl, dict) else None
    if not isinstance(gating, dict):
        if isinstance(exec_cfg, dict):
            nested = exec_cfg.get("indicator_gating")
            gating = nested if isinstance(nested, dict) else {}
        else:
            gating = {}
    if not bool(gating.get("enabled", False)):
        return 0.0
    return float(gating.get("vol_ratio_min", 0.0))


def resolve_min_validation_accuracy_gate(orch: Any | None, risk_manager: Any | None) -> float:
    """Resolve min_validation_accuracy_gate em risk_management."""
    for source in (orch, risk_manager):
        if source is None:
            continue
        config = getattr(source, "config", None)
        if not isinstance(config, dict):
            continue
        risk = config.get("risk_management")
        if isinstance(risk, dict) and risk.get("min_validation_accuracy_gate") is not None:
            return float(risk["min_validation_accuracy_gate"])
    return 0.0


def apply_microstructure_starvation_veto(
    metrics: dict,
    *,
    exec_cfg: dict | None = None,
    risk_manager: Any | None = None,
    orch: Any | None = None,
    skipped_cycles_counter: int | None = None,
) -> str | None:
    """Aplica veto duro de inanicao; retorna reason ou None se liberado."""
    skipped = resolve_skipped_cycles(skipped_cycles_counter=skipped_cycles_counter, orch=orch)
    decay = starvation_decay_factor(skipped, exec_cfg=exec_cfg if isinstance(exec_cfg, dict) else None)
    min_adx_base = resolve_min_adx_threshold(exec_cfg)
    min_adx = float(min_adx_base) * float(decay)
    metrics["quality_min_adx"] = float(min_adx)
    metrics["quality_adx_decay_factor"] = float(decay)
    if min_adx > 0.0:
        adx = _indicator_float(metrics, "adx")
        if adx is not None:
            metrics["quality_adx"] = float(adx)
            if adx + 1e-12 < min_adx:
                metrics["quality_adx_detail"] = f"adx={float(adx):.3f} < min={float(min_adx):.3f}"
                return "adx_starvation"
    min_vol_base = resolve_min_vol_ratio(orch, exec_cfg)
    min_vol = float(min_vol_base) * float(decay)
    if tcn_high_conviction_active(metrics):
        min_vol = 0.0
        metrics["vol_ratio_conviction_waiver"] = True
    metrics["quality_min_vol_ratio"] = float(min_vol)
    if min_vol > 0.0:
        vol_ratio = _indicator_float(metrics, "vol_ratio")
        if vol_ratio is None:
            vol_ratio = _indicator_float(metrics, "vol_ratio_short_long")
        if vol_ratio is not None:
            metrics["quality_vol_ratio"] = float(vol_ratio)
            if vol_ratio + 1e-12 < min_vol:
                metrics["quality_vol_ratio_detail"] = f"vol_ratio={float(vol_ratio):.3f} < min={float(min_vol):.3f}"
                return "vol_ratio_starvation"
    min_val = resolve_min_validation_accuracy_gate(orch, risk_manager)
    if min_val > 0.0:
        raw_val = metrics.get("val_accuracy")
        if raw_val is not None:
            try:
                val_accuracy = float(raw_val)
            except (TypeError, ValueError):
                val_accuracy = None
            if val_accuracy is not None and val_accuracy + 1e-12 < min_val:
                return "val_accuracy_gate"
    return None
