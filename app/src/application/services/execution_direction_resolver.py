"""Motor de direcao com refinamento de payoff continuo pelo meta-regressor LightGBM."""

from __future__ import annotations

from typing import Any

from src.application.services.direction_persistence_guard import evaluate_direction_persistence_guard
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
from src.application.services.force_trade_mode import force_trade_every_cycle
from src.application.services.live_signal_metrics import apply_live_calib_drift_soft, attach_live_signal_metrics
from src.application.services.meta_classifier_stacking import resolve_meta_payoff_edge
from src.application.services.meta_payoff_regression import apply_meta_regression_edge
from src.application.services.meta_payoff_veto_gate import (
    is_execution_signal_vetoed,
    should_veto_meta_payoff_negative_zscore,
)
from src.application.services.payoff_edge_zscore import attach_payoff_edge_zscore_metrics
from src.application.services.side_equilibrium_gate import resolve_direction_with_side_equilibrium
from src.domain.models.trade import TradeDirection


__all__ = (
    "D_SQUEEZE_BB_WIDTH_ANOMALY_RATIO",
    "_has_meta_zscore_telemetry",
    "infer_dl_direction",
    "is_technically_blocked",
    "resolve_execution_direction",
)


def _apply_persistence_guard_skip(
    entry: dict,
    metrics: dict,
    dl_dir: TradeDirection,
    *,
    symbol: str | None,
    peer_entry: dict | None,
    cycle_id: int,
    infra_cfg: dict | None,
    force: bool = False,
) -> bool:
    """Aplica persistence guard como skip; nunca aceita flip de lado."""
    if force:
        metrics.pop("persistence_guard_skip", None)
        metrics.pop("quality_guard_reject", None)
        return False
    guarded = evaluate_direction_persistence_guard(
        symbol, dl_dir, dl_dir, metrics, entry=entry, peer_entry=peer_entry, cycle_id=cycle_id, infra_cfg=infra_cfg
    )
    if guarded is None:
        metrics["gate_reason"] = str(metrics.get("gate_reason") or "persistence_guard_skip")
        metrics["persistence_guard_skip"] = True
        metrics["quality_guard_reject"] = True
        sync_entry_metrics(entry, metrics)
        return True
    return False


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
    risk_manager: Any | None = None,
    orch: Any | None = None,
    force: bool = False,
) -> tuple[TradeDirection, dict] | None:
    """Aplica decisao de execucao final."""
    if symbol is not None:
        attach_live_signal_metrics(orch, symbol, metrics)
    apply_live_calib_drift_soft(metrics, orch=orch, symbol=symbol)
    exec_dir, _final_score = apply_meta_regression_edge(
        dl_dir, metrics, predicted_edge, meta_applied=meta_applied, base_score=score, symbol=symbol
    )
    if (
        not force
        and metrics.get("quality_guard_reject")
        and str(metrics.get("gate_reason") or "") == "meta_negative_edge"
    ):
        sync_entry_metrics(entry, metrics)
        return None
    hard = should_veto_meta_payoff_negative_zscore(metrics, direction=exec_dir, risk_manager=risk_manager, orch=orch)
    if (hard or is_execution_signal_vetoed(metrics)) and not force:
        sync_entry_metrics(entry, metrics)
        return None
    if force:
        metrics.pop("signal_status", None)
        metrics["meta_veto_mode"] = "none"
        metrics["force_trade_every_cycle"] = True
    metrics.update(
        {
            "exec_direction": exec_dir.name,
            "resolved_direction": exec_dir.name,
            "tcn_score": prob,
        }
    )
    ensure_direction_margin(metrics)
    chosen = resolve_direction_with_side_equilibrium(orch, symbol, exec_dir, metrics)
    if chosen is None:
        sync_entry_metrics(entry, metrics)
        return None
    exec_dir = chosen
    metrics["exec_direction"] = exec_dir.name
    metrics["resolved_direction"] = exec_dir.name
    ensure_direction_margin(metrics)
    if float(metrics.get("side_eq_margin_boost", 0.0)) > 0.0:
        margin = float(metrics.get("direction_margin", 0.0))
        floor = float(metrics.get("quality_min_direction_margin", 0.0))
        if margin + 1e-12 < floor:
            metrics["gate_reason"] = "side_imbalance_large_n_margin"
            metrics["quality_guard_reject"] = True
            sync_entry_metrics(entry, metrics)
            return None
    if metrics.get("meta_veto_mode") is None:
        metrics["meta_veto_mode"] = "none"
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
    _ = (calibration_cfg, corr_matrix)
    exec_cfg_dict = exec_cfg if isinstance(exec_cfg, dict) else {}
    force = force_trade_every_cycle(exec_cfg_dict)
    checks = initial_direction_checks(entry, exec_cfg_dict, orch=orch)
    if checks is None:
        return None
    dl_dir, metrics, prob = checks
    if _apply_persistence_guard_skip(
        entry,
        metrics,
        dl_dir,
        symbol=symbol,
        peer_entry=peer_entry,
        cycle_id=cycle_id,
        infra_cfg=infra_cfg,
        force=force,
    ):
        return None
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
    if bool(exec_cfg_dict.get("require_meta_for_execution", False)) and not meta_applied:
        metrics["gate_reason"] = "meta_unavailable"
        metrics["quality_guard_reject"] = True
        sync_entry_metrics(entry, metrics)
        return None
    return _finalize_execution_metrics(
        entry,
        metrics,
        dl_dir,
        prob,
        predicted_edge,
        meta_applied=meta_applied,
        score=score,
        symbol=symbol,
        risk_manager=risk_manager,
        orch=orch,
        force=force,
    )
