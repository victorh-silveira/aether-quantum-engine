"""Helper functions for collect_cluster_orders in execution_collect."""

from src.application.services.execution_symbols import format_execution_alternates
from src.application.services.execution_symbols_recovery import recovery_blocked_symbols
from src.application.services.orchestrator.execution_recovery_gate import (
    recovery_min_signal,
    recovery_min_val_accuracy,
)
from src.domain.models.trade import TradeDirection


def apply_recovery_hedge_to_candidates(exec_mgr, candidates, _decisions, *, cid, mandatory=False):
    """Mantem pool de candidatos; ranking de recovery escolhe direcao e simbolo."""
    return candidates if exec_mgr and cid and mandatory is not None else candidates


def mandatory_fallback_candidates(
    exec_mgr,
    decisions,
    *,
    recovery_active,
    last_loss_symbol,
    last_loss_direction,
    skip_symbols,
    min_signal,
    min_val,
    mean_reversion=True,
    low_accuracy=True,
):
    """Monta lista com candidato forcado quando o pool DL fica vazio."""
    from src.application.services.orchestrator.execution_collect import (  # noqa: PLC0415
        build_mandatory_fallback_candidate,
    )

    fallback = build_mandatory_fallback_candidate(
        exec_mgr._trade_symbols(),
        decisions,
        recovery_active=recovery_active,
        last_loss_symbol=last_loss_symbol,
        last_loss_direction=last_loss_direction,
        skip_symbols=skip_symbols,
        min_signal=min_signal,
        min_val=min_val,
        consecutive_losses=getattr(exec_mgr.orch.risk_manager, "consecutive_losses", 0),
        mean_reversion_enabled=mean_reversion,
        low_accuracy_enabled=low_accuracy,
    )
    return [fallback] if fallback else []


def mandatory_fallback_if_empty(
    exec_mgr,
    decisions,
    candidates,
    *,
    mandatory,
    recovery_active,
    last_loss,
    last_loss_dir,
    skip_symbols,
    min_signal,
    min_val,
    mean_reversion=True,
    low_accuracy=True,
):
    """Aplica fallback obrigatorio quando o pool de candidatos DL fica vazio."""
    if candidates or not mandatory:
        return candidates
    return mandatory_fallback_candidates(
        exec_mgr,
        decisions,
        recovery_active=recovery_active,
        last_loss_symbol=last_loss,
        last_loss_direction=last_loss_dir,
        skip_symbols=skip_symbols,
        min_signal=min_signal,
        min_val=min_val,
        mean_reversion=mean_reversion,
        low_accuracy=low_accuracy,
    )


def extract_collect_params(exec_mgr, dl_cfg: dict, *, recovery_active: bool) -> tuple:
    """Extrai parametros de configuracao necessarios para collect_cluster_orders."""
    recovery_cfg = dl_cfg.get("recovery_gating", {}) if isinstance(dl_cfg, dict) else {}
    risk_cfg = exec_mgr.orch.config.get("risk_management", {}) if isinstance(exec_mgr.orch.config, dict) else {}
    kelly_cfg = risk_cfg.get("kelly", {}) if isinstance(risk_cfg, dict) else {}
    proposal_skip_fn = getattr(exec_mgr.orch.risk_manager, "proposal_skip_symbols", None)
    proposal_skip = proposal_skip_fn() if callable(proposal_skip_fn) else frozenset()
    recovery_skip = recovery_blocked_symbols(exec_mgr.orch.risk_manager, kelly_cfg) if recovery_active else frozenset()
    skip_symbols = proposal_skip | recovery_skip
    pending_total = sum(float(v) for v in getattr(exec_mgr.orch.risk_manager, "pending_loss", {}).values())
    consecutive_losses = getattr(exec_mgr.orch.risk_manager, "consecutive_losses", 0)
    min_signal = recovery_min_signal(
        kelly_cfg,
        recovery_active=recovery_active,
        pending_total=pending_total,
        consecutive_losses=consecutive_losses,
    )
    min_val = (
        recovery_min_val_accuracy(kelly_cfg, consecutive_losses=consecutive_losses)
        if recovery_active
        else float(dl_cfg.get("min_val_accuracy", 0.54))
    )
    min_edge = float(dl_cfg.get("min_edge_execute", 0.0))
    last_loss = getattr(exec_mgr.orch.risk_manager, "last_loss_symbol", None)
    last_loss_dir = getattr(exec_mgr.orch.risk_manager, "last_loss_direction", None)
    exec_cfg = exec_mgr.orch.config.get("orchestrator", {}).get("execution", {})
    mean_reversion = bool(exec_cfg.get("mean_reversion_inversion_enabled", True))
    low_accuracy = bool(exec_cfg.get("low_accuracy_inversion_enabled", True))
    return (
        recovery_cfg,
        kelly_cfg,
        skip_symbols,
        min_signal,
        min_val,
        min_edge,
        last_loss,
        last_loss_dir,
        mean_reversion,
        low_accuracy,
        exec_cfg,
    )


def log_execution_decision(exec_mgr, cid: str, best: tuple, candidates: list, effective_signal: float) -> None:
    """Registra log detalhado da decisao de execucao e indicadores."""
    metrics = best[2]
    alts_str = format_execution_alternates(candidates, exclude_symbol=best[0])
    alt_suffix = f" | alt={alts_str}" if alts_str else ""
    indicators = metrics.get("indicators", {})
    ind_str = " | ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in indicators.items())
    raw_val = float(metrics.get("raw_prob", 0.5))
    exec_mgr.logger.info(
        "[%s] EXEC_SEL | %s ord=%s dl=%s s=%.2f v=%.2f r=%.2f | P(CALL)=%.2f P(PUT)=%.2f | Acc=%.2f Score=%.2f | Votes: CALL=%d PUT=%d | %s%s",
        cid,
        best[0],
        best[1].name,
        metrics.get("dl_direction", best[1].name),
        effective_signal,
        float(metrics.get("val_accuracy", 0.0)),
        raw_val if best[1] == TradeDirection.CALL else 1.0 - raw_val,
        raw_val,
        1.0 - raw_val,
        float(metrics.get("val_accuracy", 0.0)),
        effective_signal,
        metrics.get("call_votes", 0),
        metrics.get("put_votes", 0),
        ind_str,
        alt_suffix,
    )
