"""Coleta de candidatos com boletamento continuo, sem veto de qualidade."""

from src.application.services.execution_direction import build_execution_candidate
from src.application.services.execution_direction_cross_corr import cached_correlation_matrix
from src.application.services.execution_quality_gate import apply_quality_penalty_to_metrics
from src.application.services.execution_volatility_booster import apply_volatility_vol_booster
from src.application.services.orchestrator.execution_recovery_gate import cluster_entry_eligible


def gather_cluster_candidates(
    exec_mgr,
    decisions,
    *,
    recovery_active,
    recovery_cfg,
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
            recovery_cfg=recovery_cfg,
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
        )
        if built is None:
            continue
        _, _, metrics = built
        apply_volatility_vol_booster(
            metrics,
            mandatory_min_trade_score=min_signal,
            min_edge_execute=min_edge,
        )
        apply_quality_penalty_to_metrics(metrics)
        candidates.append(built)
    return candidates
