"""Orchestrator service for collecting and selecting cluster execution candidates."""

__all__ = ["collect_cluster_orders", "_mandatory_fallback_candidates"]

from src.application.services.execution_direction import build_execution_candidate
from src.application.services.execution_direction_fallback import build_mandatory_fallback_candidate
from src.application.services.execution_mandatory_pick import pick_absolute_mandatory_candidate
from src.application.services.execution_market_rank import build_market_execution_candidate
from src.application.services.execution_symbols import (
    filter_execution_candidates,
    select_best_execution_candidate,
    select_mandatory_execution_candidate,
)
from src.application.services.execution_symbols_recovery import (
    apply_recovery_direction_flip,
    pending_recovery_active,
)
from src.application.services.log_dedupe import log_info_if_changed
from src.application.services.orchestrator.execution_collect_helpers import (
    apply_recovery_hedge_to_candidates,
    extract_collect_params,
    log_execution_decision,
    mandatory_fallback_candidates as _mandatory_fallback_candidates,  # noqa: F401
    mandatory_fallback_if_empty,
)
from src.application.services.orchestrator.execution_recovery_gate import (
    cluster_entry_eligible,
)
from src.domain.models.trade import TradeDirection
from src.domain.risk.stake_sizing import enrich_metrics_conviction, raw_side_from_metrics


def _gather_cluster_candidates(
    exec_mgr,
    decisions,
    *,
    mandatory,
    recovery_active,
    recovery_cfg,
    cid,
    min_signal,
    min_val,
    mean_reversion=True,
    low_accuracy=True,
):
    """Coleta candidatos DL elegiveis para o ciclo atual."""
    candidates = []
    for symbol in exec_mgr._trade_symbols():
        entry = decisions.get(symbol)
        if not entry:
            continue
        if not cluster_entry_eligible(
            entry,
            mandatory=mandatory,
            recovery_active=recovery_active,
            recovery_cfg=recovery_cfg,
            min_signal=min_signal,
            min_val=min_val,
        ):
            exec_mgr.logger.debug("[%s] SKIP: Conviccao insuficiente para %s (Metrics Gate)", cid, symbol)
            continue
        built = (
            build_market_execution_candidate(
                symbol,
                entry,
                recovery_active=recovery_active,
                consecutive_losses=getattr(exec_mgr.orch.risk_manager, "consecutive_losses", 0),
                mean_reversion_enabled=mean_reversion,
                low_accuracy_enabled=low_accuracy,
            )
            if mandatory
            else build_execution_candidate(symbol, entry)
        )
        if built is not None:
            candidates.append(built)
    return candidates


def _select_cluster_best(exec_mgr, candidates, *, mandatory, last_loss, last_loss_dir, recovery_active, skip_symbols):
    """Filtra e escolhe o melhor candidato do cluster para execucao no ciclo."""
    if not mandatory:
        dl_cfg = exec_mgr.orch.config.get("deep_learning", {})
        selection = dl_cfg.get("selection", {}) if isinstance(dl_cfg, dict) else {}
        candidates = filter_execution_candidates(candidates, selection=selection)
        if not candidates:
            return None
    exec_cfg = exec_mgr.orch.config.get("orchestrator", {}).get("execution", {})
    margin = float(exec_cfg.get("diversify_after_loss_margin", 0.08))
    if mandatory:
        return select_mandatory_execution_candidate(
            exec_mgr.orch,
            candidates,
            last_loss_symbol=last_loss,
            last_loss_direction=last_loss_dir,
            diversify_margin=margin,
            recovery_active=recovery_active,
            skip_symbols=skip_symbols,
        )
    return select_best_execution_candidate(
        candidates,
        last_loss_symbol=last_loss,
        last_loss_direction=last_loss_dir,
        diversify_margin=margin,
        recovery_active=recovery_active,
    )


def collect_cluster_orders(exec_mgr, decisions: dict) -> list[tuple[str, TradeDirection, dict]]:
    """Seleciona uma ordem por ciclo; modo obrigatorio ignora gate execute=false."""
    mandatory = exec_mgr._mandatory_trade_each_cycle()
    recovery_active = pending_recovery_active(getattr(exec_mgr.orch.risk_manager, "pending_loss", {}))
    dl_cfg = exec_mgr.orch.config.get("deep_learning", {})
    (
        recovery_cfg,
        kelly_cfg,
        skip_symbols,
        min_signal,
        min_val,
        last_loss,
        last_loss_dir,
        mean_reversion,
        low_accuracy,
        exec_cfg,
    ) = extract_collect_params(exec_mgr, dl_cfg, recovery_active=recovery_active)
    cid = f"C{int(exec_mgr.orch._active_cycle_id):04d}"

    candidates = _gather_cluster_candidates(
        exec_mgr,
        decisions,
        mandatory=mandatory,
        recovery_active=recovery_active,
        recovery_cfg=recovery_cfg,
        cid=cid,
        min_signal=min_signal,
        min_val=min_val,
        mean_reversion=mean_reversion,
        low_accuracy=low_accuracy,
    )
    candidates = mandatory_fallback_if_empty(
        exec_mgr,
        decisions,
        candidates,
        mandatory=mandatory,
        recovery_active=recovery_active,
        last_loss=last_loss,
        last_loss_dir=last_loss_dir,
        skip_symbols=skip_symbols,
        min_signal=min_signal,
        min_val=min_val,
        mean_reversion=mean_reversion,
        low_accuracy=low_accuracy,
    )
    if candidates and skip_symbols:
        candidates = [item for item in candidates if item[0] not in skip_symbols]
    if candidates:
        candidates = apply_recovery_hedge_to_candidates(exec_mgr, candidates, decisions, cid=cid, mandatory=mandatory)
        candidates = mandatory_fallback_if_empty(
            exec_mgr,
            decisions,
            candidates,
            mandatory=mandatory,
            recovery_active=recovery_active,
            last_loss=last_loss,
            last_loss_dir=last_loss_dir,
            skip_symbols=skip_symbols,
            min_signal=min_signal,
            min_val=min_val,
            mean_reversion=mean_reversion,
            low_accuracy=low_accuracy,
        )
    best = _select_cluster_best(
        exec_mgr,
        candidates,
        mandatory=mandatory,
        last_loss=last_loss,
        last_loss_dir=last_loss_dir,
        recovery_active=recovery_active,
        skip_symbols=skip_symbols,
    )
    if best is None and mandatory:
        ultimate = build_mandatory_fallback_candidate(
            exec_mgr._trade_symbols(),
            decisions,
            recovery_active=recovery_active,
            last_loss_symbol=last_loss,
            last_loss_direction=last_loss_dir,
            skip_symbols=skip_symbols,
            min_signal=min_signal,
            min_val=min_val,
            consecutive_losses=getattr(exec_mgr.orch.risk_manager, "consecutive_losses", 0),
            mean_reversion_enabled=mean_reversion,
            low_accuracy_enabled=low_accuracy,
        )
        if ultimate is None:
            ultimate = pick_absolute_mandatory_candidate(
                exec_mgr._trade_symbols(),
                decisions,
                recovery_active=recovery_active,
                last_loss_symbol=last_loss,
                last_loss_direction=last_loss_dir,
                min_signal=min_signal,
                min_val=min_val,
                mean_reversion_enabled=mean_reversion,
                low_accuracy_enabled=low_accuracy,
            )
            if ultimate is None:
                ultimate = pick_absolute_mandatory_candidate(
                    exec_mgr._trade_symbols(),
                    decisions,
                    recovery_active=recovery_active,
                    last_loss_symbol=last_loss,
                    last_loss_direction=last_loss_dir,
                    min_signal=0.0,
                    min_val=0.0,
                    mean_reversion_enabled=mean_reversion,
                    low_accuracy_enabled=low_accuracy,
                )
        if ultimate is not None:
            best = ultimate
            candidates = [ultimate]
    exec_cfg = exec_mgr.orch.config.get("orchestrator", {}).get("execution", {})
    best = apply_recovery_direction_flip(
        best,
        decisions,
        recovery_active=recovery_active,
        last_loss_symbol=last_loss,
        last_loss_direction=last_loss_dir,
        flip_enabled=bool(exec_cfg.get("recovery_flip_direction_after_loss", True)),
        flip_max_conviction=float(exec_cfg.get("recovery_flip_max_conviction", 0.56)),
        consecutive_losses=getattr(exec_mgr.orch.risk_manager, "consecutive_losses", 0),
        flip_use_trend=bool(exec_cfg.get("recovery_flip_use_trend_confirmation", False)),
    )
    if best is not None:
        metrics = best[2]
        min_raw = float(kelly_cfg.get("stake_conviction_min_raw", 0.51))
        enrich_metrics_conviction(metrics, min_raw=min_raw)
        calibrated = float(metrics.get("trade_score", metrics.get("conviction", 0.0)))
        raw_side = raw_side_from_metrics(metrics)
        effective_signal = max(calibrated, raw_side)
        log_execution_decision(exec_mgr, cid, best, candidates, effective_signal)
        return [best]

    grey_zone_symbols = []
    for symbol, entry in decisions.items():
        metrics = entry.get("metrics") or {}
        val = float(metrics.get("val_accuracy", 0.50))
        if 0.46 <= val < 0.50:
            grey_zone_symbols.append(f"{symbol}(v={val:.2f})")
    if grey_zone_symbols:
        log_info_if_changed(
            exec_mgr.orch,
            exec_mgr.logger,
            "exec_skip_grey",
            ", ".join(grey_zone_symbols),
            "[%s] EXEC_SKIP || %s | acuracia na zona cinza [0.46, 0.50) | ignorando execucao obrigatoria por seguranca",
            cid,
            ", ".join(grey_zone_symbols),
        )
    return []
