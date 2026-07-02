"""Coleta de candidatos sem veto de qualidade operacional."""

from src.application.services.deep_learning.dl_params import parse_dynamic_threshold_config
from src.application.services.execution_direction import build_execution_candidate
from src.application.services.execution_direction_cross_corr import cached_correlation_matrix
from src.application.services.execution_quality_gate import apply_quality_penalty_to_metrics, quality_gate_params
from src.application.services.execution_universal_regime_gate import log_regime_audit
from src.application.services.execution_volatility_booster import apply_volatility_vol_booster
from src.application.services.orchestrator.execution_recovery_gate import cluster_entry_eligible
from src.domain.models.trade import TradeDirection


def _dl_direction_from_metrics(metrics: dict, fallback: TradeDirection) -> TradeDirection:
    """Resolve direcao DL das metricas com fallback para ordem resolvida."""
    raw = metrics.get("dl_direction")
    if raw is None:
        return fallback
    try:
        return TradeDirection[str(raw).upper()]
    except (KeyError, ValueError):
        return fallback


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
    """Coleta candidatos DL elegiveis aplicando penalidade em vez de veto de qualidade."""
    exec_cfg = exec_mgr.orch.config.get("orchestrator", {}).get("execution", {})
    calibration_cfg = exec_mgr.orch.config.get("deep_learning", {}).get("calibration")
    kelly = kelly_cfg if isinstance(kelly_cfg, dict) else {}
    qparams = quality_gate_params(exec_cfg)
    dynamic_cfg = parse_dynamic_threshold_config(exec_cfg if isinstance(exec_cfg, dict) else {})
    exhaustion_gate = exec_cfg.get("exhaustion_gate") if isinstance(exec_cfg.get("exhaustion_gate"), dict) else {}
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
        )
        if built is None:
            continue
        _, direction, metrics = built
        if metrics.get("regime_skip_cycle"):
            continue
        dl_dir = _dl_direction_from_metrics(metrics, direction)
        log_regime_audit(exec_mgr.logger, cid, symbol, dl_dir, direction, metrics, recovery_active=recovery_active)
        boosted_signal, boosted_edge = apply_volatility_vol_booster(
            metrics,
            mandatory_min_trade_score=min_signal,
            min_edge_execute=min_edge,
        )
        apply_quality_penalty_to_metrics(
            metrics,
            min_signal=boosted_signal,
            min_val=min_val,
            min_edge=boosted_edge,
            recovery_active=recovery_active,
            dynamic_threshold_cfg=dynamic_cfg,
            exhaustion_gate_cfg=exhaustion_gate,
            recovery_kelly_cfg=kelly if recovery_active else None,
            consecutive_losses=consecutive_losses,
            recovery_skip_counter=recovery_skip_counter,
            session_drawdown=session_drawdown,
            **qparams,
        )
        candidates.append(built)
    return candidates
