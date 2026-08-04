"""Coleta de candidatos com boletamento continuo, sem veto de qualidade."""

from src.application.services.execution_direction import build_execution_candidate
from src.application.services.execution_direction_cross_corr import cached_correlation_matrix
from src.application.services.execution_loss_protection import apply_loss_protection_penalties
from src.application.services.execution_quality_gate import apply_quality_penalty_to_metrics
from src.application.services.execution_volatility_booster import apply_volatility_vol_booster
from src.application.services.orchestrator.execution_recovery_gate import cluster_entry_eligible


def _sync_entry_metrics(entry: dict, metrics: dict) -> None:
    """Propaga flags de suspensao do candidato construido de volta ao entry."""
    entry_metrics = entry.get("metrics")
    if not isinstance(entry_metrics, dict):
        entry["metrics"] = dict(metrics)
        return
    for key in (
        "signal_status",
        "regime_guard_action",
        "quality_guard_reject",
        "regime_skip_cycle",
        "gate_reason",
        "quality_gate_reason",
        "side_eq_blocked",
        "execution_candidate_ready",
        "side_eq_action",
        "side_eq_reason",
        "price_zone",
        "meta_veto_mode",
        "exec_direction",
        "resolved_direction",
    ):
        if key in metrics:
            entry_metrics[key] = metrics[key]


def gather_cluster_candidates(
    exec_mgr,
    decisions,
    *,
    recovery_active,
    cid,
    min_signal,
    min_val,
    min_edge=0.0,
    kelly_cfg=None,
    consecutive_losses=0,
    recovery_skip_counter=0,
    session_drawdown=0.0,
):
    """Coleta candidatos DL elegiveis; qualquer sinal valido participa do pool."""
    _ = (cid, kelly_cfg, consecutive_losses, recovery_skip_counter, session_drawdown)
    exec_cfg = exec_mgr.orch.config.get("orchestrator", {}).get("execution", {})
    calibration_cfg = exec_mgr.orch.config.get("deep_learning", {}).get("calibration")
    infra_cfg = exec_mgr.orch.config.get("infra", {})
    corr_matrix = cached_correlation_matrix(exec_mgr.orch)
    candidates = []
    for symbol in exec_mgr._trade_symbols():
        entry = decisions.get(symbol)
        if not entry:
            continue
        if not cluster_entry_eligible(
            entry,
            mandatory=True,
            recovery_active=recovery_active,
            min_signal=min_signal,
            min_val=min_val,
            min_edge=min_edge,
        ):
            continue
        built = build_execution_candidate(
            symbol,
            entry,
            exec_cfg=exec_cfg,
            calibration_cfg=calibration_cfg,
            recovery_active=recovery_active,
            corr_matrix=corr_matrix or None,
            infra_cfg=infra_cfg if isinstance(infra_cfg, dict) else None,
            decisions=decisions,
            cycle_id=int(exec_mgr.orch._active_cycle_id),
            risk_manager=getattr(exec_mgr.orch, "risk_manager", None),
            skipped_cycles_counter=int(getattr(exec_mgr.orch, "_quality_skipped_cycles_counter", 0) or 0),
            orch=exec_mgr.orch,
        )
        if built is None:
            continue
        _, _, metrics = built
        apply_volatility_vol_booster(
            metrics,
            mandatory_min_trade_score=min_signal,
            min_edge_execute=min_edge,
        )
        apply_quality_penalty_to_metrics(
            metrics,
            exec_cfg=exec_cfg if isinstance(exec_cfg, dict) else {},
            risk_manager=getattr(exec_mgr.orch, "risk_manager", None),
            skipped_cycles_counter=int(getattr(exec_mgr.orch, "_quality_skipped_cycles_counter", 0) or 0),
            orch=exec_mgr.orch,
        )
        apply_loss_protection_penalties(metrics, exec_direction=built[1])
        if metrics.get("signal_status") == "SKIP":
            _sync_entry_metrics(entry, metrics)
            continue
        _sync_entry_metrics(entry, metrics)
        candidates.append(built)
    return candidates
