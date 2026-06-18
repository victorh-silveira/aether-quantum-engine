"""Montagem e selecao de ordens do cluster para o ExecutionManager."""

from src.application.services.execution_direction import (
    build_execution_candidate,
)
from src.application.services.execution_direction_fallback import build_mandatory_fallback_candidate
from src.application.services.execution_mandatory_pick import pick_absolute_mandatory_candidate
from src.application.services.execution_market_rank import build_market_execution_candidate
from src.application.services.execution_symbols import (
    filter_execution_candidates,
    format_execution_alternates,
    select_best_execution_candidate,
    select_mandatory_execution_candidate,
)
from src.application.services.execution_symbols_recovery import (
    apply_recovery_direction_flip,
    pending_recovery_active,
    recovery_blocked_symbols,
)
from src.application.services.orchestrator.execution_near_stop_win import should_pause_weak_mandatory
from src.application.services.orchestrator.execution_recovery_gate import (
    cluster_entry_eligible,
    recovery_min_signal,
    recovery_min_val_accuracy,
)
from src.domain.models.trade import TradeDirection
from src.domain.risk.stake_sizing import enrich_metrics_conviction, raw_side_from_metrics


def apply_recovery_hedge_to_candidates(
    exec_mgr, candidates: list[tuple[str, TradeDirection, dict]], _decisions: dict, *, cid: str, mandatory: bool = False
) -> list[tuple[str, TradeDirection, dict]]:
    """Mantem pool de candidatos; ranking de recovery escolhe direcao e simbolo."""
    _ = (exec_mgr, cid, mandatory)
    return candidates


def _mandatory_fallback_candidates(
    exec_mgr,
    decisions: dict,
    *,
    recovery_active: bool,
    last_loss_symbol: str | None,
    last_loss_direction: str | None,
    skip_symbols: frozenset[str],
    min_signal: float,
    min_val: float,
) -> list[tuple[str, TradeDirection, dict]]:
    """Monta lista com candidato forcado quando o pool DL fica vazio."""
    fallback = build_mandatory_fallback_candidate(
        exec_mgr._trade_symbols(),
        decisions,
        recovery_active=recovery_active,
        last_loss_symbol=last_loss_symbol,
        last_loss_direction=last_loss_direction,
        skip_symbols=skip_symbols,
        min_signal=min_signal,
        min_val=min_val,
    )
    return [fallback] if fallback else []


def _gather_cluster_candidates(
    exec_mgr,
    decisions: dict,
    *,
    mandatory: bool,
    recovery_active: bool,
    recovery_cfg: dict,
    cid: str,
    min_signal: float,
    min_val: float,
) -> list[tuple[str, TradeDirection, dict]]:
    """Coleta candidatos DL elegiveis para o ciclo atual."""
    candidates: list[tuple[str, TradeDirection, dict]] = []
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
            build_market_execution_candidate(symbol, entry) if mandatory else build_execution_candidate(symbol, entry)
        )
        if built is None:
            continue
        candidates.append(built)
    return candidates


def _mandatory_fallback_if_empty(
    exec_mgr,
    decisions: dict,
    candidates: list[tuple[str, TradeDirection, dict]],
    *,
    mandatory: bool,
    recovery_active: bool,
    last_loss: str | None,
    last_loss_dir: str | None,
    skip_symbols: frozenset[str],
    min_signal: float,
    min_val: float,
) -> list[tuple[str, TradeDirection, dict]]:
    """Aplica fallback obrigatorio quando o pool de candidatos DL fica vazio."""
    if candidates or not mandatory:
        return candidates
    return _mandatory_fallback_candidates(
        exec_mgr,
        decisions,
        recovery_active=recovery_active,
        last_loss_symbol=last_loss,
        last_loss_direction=last_loss_dir,
        skip_symbols=skip_symbols,
        min_signal=min_signal,
        min_val=min_val,
    )


def _select_cluster_best(
    exec_mgr,
    candidates: list[tuple[str, TradeDirection, dict]],
    *,
    mandatory: bool,
    last_loss: str | None,
    last_loss_dir: str | None,
    recovery_active: bool,
    skip_symbols: frozenset[str],
) -> tuple[str, TradeDirection, dict] | None:
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
    recovery_cfg = dl_cfg.get("recovery_gating", {}) if isinstance(dl_cfg, dict) else {}
    risk_cfg = exec_mgr.orch.config.get("risk_management", {}) if isinstance(exec_mgr.orch.config, dict) else {}
    kelly_cfg = risk_cfg.get("kelly", {}) if isinstance(risk_cfg, dict) else {}
    proposal_skip_fn = getattr(exec_mgr.orch.risk_manager, "proposal_skip_symbols", None)
    proposal_skip = proposal_skip_fn() if callable(proposal_skip_fn) else frozenset()
    recovery_skip = recovery_blocked_symbols(exec_mgr.orch.risk_manager, kelly_cfg) if recovery_active else frozenset()
    skip_symbols = proposal_skip | recovery_skip
    pending_total = sum(float(v) for v in getattr(exec_mgr.orch.risk_manager, "pending_loss", {}).values())
    min_signal = recovery_min_signal(kelly_cfg, recovery_active=recovery_active, pending_total=pending_total)
    min_val = recovery_min_val_accuracy(kelly_cfg) if recovery_active else 0.0
    cid = f"C{int(exec_mgr.orch._active_cycle_id):04d}"
    last_loss = getattr(exec_mgr.orch.risk_manager, "last_loss_symbol", None)
    last_loss_dir = getattr(exec_mgr.orch.risk_manager, "last_loss_direction", None)
    orders: list[tuple[str, TradeDirection, dict]] = []
    candidates = _gather_cluster_candidates(
        exec_mgr,
        decisions,
        mandatory=mandatory,
        recovery_active=recovery_active,
        recovery_cfg=recovery_cfg,
        cid=cid,
        min_signal=min_signal,
        min_val=min_val,
    )
    candidates = _mandatory_fallback_if_empty(
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
    )
    if candidates and skip_symbols:
        candidates = [item for item in candidates if item[0] not in skip_symbols]
    if candidates:
        candidates = apply_recovery_hedge_to_candidates(exec_mgr, candidates, decisions, cid=cid, mandatory=mandatory)
        candidates = _mandatory_fallback_if_empty(
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
            )
        if ultimate is None and not should_pause_weak_mandatory(exec_mgr, decisions, recovery_active=recovery_active):
            ultimate = pick_absolute_mandatory_candidate(
                exec_mgr._trade_symbols(),
                decisions,
                recovery_active=recovery_active,
                last_loss_symbol=last_loss,
                last_loss_direction=last_loss_dir,
                min_signal=0.0,
                min_val=0.0,
            )
        if ultimate is not None:
            best = ultimate
            candidates = [ultimate]
    exec_cfg = exec_mgr.orch.config.get("orchestrator", {}).get("execution", {})
    flip_recovery = bool(exec_cfg.get("recovery_flip_direction_after_loss", True))
    flip_max_conviction = float(exec_cfg.get("recovery_flip_max_conviction", 0.56))
    best = apply_recovery_direction_flip(
        best,
        decisions,
        recovery_active=recovery_active,
        last_loss_symbol=last_loss,
        last_loss_direction=last_loss_dir,
        flip_enabled=flip_recovery,
        flip_max_conviction=flip_max_conviction,
        consecutive_losses=getattr(exec_mgr.orch.risk_manager, "consecutive_losses", 0),
    )
    if best is not None:
        metrics = best[2]
        min_raw = float(kelly_cfg.get("stake_conviction_min_raw", 0.51))
        enrich_metrics_conviction(metrics, min_raw=min_raw)
        inv_tag = " inv" if metrics.get("direction_inverted") and not metrics.get("recovery_forced") else ""
        dl_name = metrics.get("dl_direction", best[1].name)
        calibrated = float(metrics.get("trade_score", metrics.get("conviction", 0.0)))
        raw_side = raw_side_from_metrics(metrics)
        effective_signal = max(calibrated, raw_side)
        exec_mgr.logger.info(
            "[%s] EXEC_SEL | %s ord=%s dl=%s%s s=%.2f v=%.2f r=%.2f | alt=%s",
            cid,
            best[0],
            best[1].name,
            dl_name,
            inv_tag,
            effective_signal,
            float(metrics.get("val_accuracy", 0.0)),
            float(metrics.get("raw_prob", metrics.get("raw_conviction", 0.0))),
            format_execution_alternates(candidates, exclude_symbol=best[0]),
        )
        orders = [best]
    return orders
