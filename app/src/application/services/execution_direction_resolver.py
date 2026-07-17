"""Motor de direcao com refinamento de payoff continuo pelo meta-regressor LightGBM."""

from __future__ import annotations

from typing import Any

from src.application.services.execution_direction_checks import (
    D_SQUEEZE_BB_WIDTH_ANOMALY_RATIO,
    has_meta_zscore_telemetry as _has_meta_zscore_telemetry,
    infer_dl_direction,
    initial_direction_checks,
    is_technically_blocked,
    reject_on_quality_gate,
    seed_direction_metrics,
    sync_entry_metrics,
)
from src.application.services.execution_quality_gate import ensure_direction_margin
from src.application.services.meta_classifier_stacking import resolve_meta_payoff_edge
from src.application.services.meta_payoff_regression import apply_meta_regression_edge
from src.application.services.payoff_edge_zscore import attach_payoff_edge_zscore_metrics
from src.domain.models.trade import TradeDirection


__all__ = (
    "D_SQUEEZE_BB_WIDTH_ANOMALY_RATIO",
    "_has_meta_zscore_telemetry",
    "infer_dl_direction",
    "is_technically_blocked",
    "resolve_execution_direction",
)


def _finalize_execution_metrics(
    entry: dict,
    metrics: dict,
    dl_dir: TradeDirection,
    prob: float,
    predicted_edge: float,
    *,
    meta_applied: bool,
    score: float,
    symbol: str | None,
) -> tuple[TradeDirection, dict]:
    """Aplica decisao de execucao final."""
    exec_dir, _final_score = apply_meta_regression_edge(
        dl_dir,
        metrics,
        predicted_edge,
        meta_applied=meta_applied,
        base_score=score,
        symbol=symbol,
    )
    metrics.update(
        {
            "exec_direction": exec_dir.name,
            "resolved_direction": exec_dir.name,
            "direction_inverted": exec_dir != dl_dir,
            "tcn_score": prob,
        }
    )
    ensure_direction_margin(metrics)
    sync_entry_metrics(entry, metrics)
    return exec_dir, metrics


def resolve_execution_direction(
    entry: dict,
    *,
    exec_cfg: dict | None = None,
    calibration_cfg: dict | None = None,
    recovery_active: bool = False,
    symbol: str | None = None,
    corr_matrix: dict[tuple[str, str], float] | None = None,
    infra_cfg: dict | None = None,
    peer_entry: dict | None = None,
    cycle_id: int = 0,
    risk_manager: Any | None = None,
    skipped_cycles_counter: int | None = None,
    orch: Any | None = None,
) -> tuple[TradeDirection, dict] | None:
    """Resolve direcao micro fiel ao sinal TCN/DL com telemetria meta-regressor."""
    _ = (calibration_cfg, corr_matrix, peer_entry, cycle_id)
    exec_cfg_dict = exec_cfg if isinstance(exec_cfg, dict) else {}
    checks = initial_direction_checks(entry, exec_cfg_dict, orch=orch)
    if checks is None:
        return None
    dl_dir, metrics, prob = checks
    score = seed_direction_metrics(metrics, dl_dir=dl_dir, prob=prob)
    predicted_edge, meta_applied = resolve_meta_payoff_edge(
        symbol=symbol,
        metrics=metrics,
        direction=dl_dir,
        tcn_probability=prob,
        _base_score=score,
        config={"infra": infra_cfg} if infra_cfg else None,
    )
    if metrics.get("meta_payoff_edge_zscore") is None and metrics.get("edge_zscore") is None:
        attach_payoff_edge_zscore_metrics(
            metrics, float(metrics.get("predicted_payoff_edge", predicted_edge)), symbol=symbol
        )
    gate_probe = dict(metrics)
    gate_probe["predicted_payoff_edge"] = float(predicted_edge)
    gate_probe["meta_classifier_applied"] = bool(meta_applied)
    kw = {
        "risk_manager": risk_manager,
        "recovery_active": recovery_active,
        "skipped_cycles_counter": skipped_cycles_counter,
        "orch": orch,
    }
    if reject_on_quality_gate(entry, metrics, gate_probe, exec_cfg_dict, **kw):
        return None
    _ = bool(exec_cfg_dict.get("require_meta_for_execution", False))
    return _finalize_execution_metrics(
        entry,
        metrics,
        dl_dir,
        prob,
        predicted_edge,
        meta_applied=meta_applied,
        score=score,
        symbol=symbol,
    )
