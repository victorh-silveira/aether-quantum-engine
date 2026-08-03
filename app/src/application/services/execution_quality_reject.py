"""Rejeicao dura por quality gate, ACC e margem calibrada fraca."""

from __future__ import annotations

from typing import Any

from src.application.services.doctrine_invariants import resolve_hard_cal_margin_floor
from src.application.services.execution_quality_gate import passes_execution_quality
from src.application.services.execution_quality_gate_meta import evaluate_meta_payoff_quality
from src.application.services.force_trade_mode import force_trade_every_cycle
from src.domain.risk.stake_sizing import metric_float


_HARD_QUALITY_REASONS = frozenset({"val_accuracy_gate", "adx_starvation", "vol_ratio_starvation"})


def has_meta_zscore_telemetry(metrics: dict) -> bool:
    """True quando ha Z-Score meta com amostras suficientes."""
    if metrics.get("meta_payoff_edge_zscore") is None and metrics.get("edge_zscore") is None:
        return False
    s = metrics.get("edge_zscore_samples")
    return s is None or int(s) >= 2


def reject_on_quality_gate(
    entry: dict,
    metrics: dict,
    gate_probe: dict,
    exec_cfg_dict: dict,
    *,
    risk_manager: Any | None = None,
    recovery_active: bool = False,
    skipped_cycles_counter: int | None = None,
    orch: Any | None = None,
) -> bool:
    """Rejeita por margem calibrada fraca, ACC baixo ou quality gate duro."""
    _ = entry
    if force_trade_every_cycle(exec_cfg_dict):
        for k in ("quality_guard_reject", "regime_skip_cycle", "quality_gate_reason"):
            gate_probe.pop(k, None)
            metrics.pop(k, None)
        return False
    kw = {
        "exec_cfg": exec_cfg_dict,
        "risk_manager": risk_manager,
        "skipped_cycles_counter": skipped_cycles_counter,
        "orch": orch,
    }
    if has_meta_zscore_telemetry(gate_probe):
        evaluate_meta_payoff_quality(gate_probe, **kw)
    passed = passes_execution_quality(gate_probe, **kw)
    for k in (
        "execution_gate_state",
        "quality_gate_regime",
        "direction_margin",
        "quality_min_direction_margin",
        "quality_min_payoff_edge",
        "quality_skipped_cycles_counter",
        "quality_starvation_decay_factor",
        "quality_gate_reason",
    ):
        if k in gate_probe:
            metrics[k] = gate_probe[k]
    if not passed:
        reason = str(gate_probe.get("quality_gate_reason") or "")
        if reason in _HARD_QUALITY_REASONS:
            metrics["quality_guard_reject"] = True
            metrics["regime_skip_cycle"] = True
            metrics["quality_gate_reason"] = reason
            metrics["gate_reason"] = reason
            return True
    hard = resolve_hard_cal_margin_floor(exec_cfg_dict)
    if hard > 0.0 and not recovery_active:
        senior = metric_float(metrics, "senior_trader_conviction", default=0.0)
        waived = senior + 1e-12 >= 0.56
        if not waived:
            from src.application.services.execution_quality_gate_config import (  # noqa: PLC0415
                resolve_quality_gate_config,
            )

            qg = resolve_quality_gate_config(exec_cfg_dict)
            z_min = float(qg.get("min_meta_payoff_zscore", 0.0) or 0.0)
            z_raw = metrics.get("meta_payoff_edge_zscore", metrics.get("edge_zscore"))
            if z_min > 0.0 and z_raw is not None:
                try:
                    waived = float(z_raw) + 1e-12 >= z_min
                except (TypeError, ValueError):
                    waived = False
        if not waived:
            prob = metrics.get("calibrated_prob", metrics.get("raw_prob"))
            if prob is not None and abs(float(prob) - 0.5) + 1e-12 < hard:
                metrics["quality_guard_reject"] = True
                metrics["regime_skip_cycle"] = True
                metrics["quality_gate_reason"] = "cal_margin_floor"
                metrics["gate_reason"] = "cal_margin_floor"
                return True
    metrics.pop("quality_guard_reject", None)
    metrics.pop("regime_skip_cycle", None)
    return False
