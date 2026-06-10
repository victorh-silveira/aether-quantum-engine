"""Montagem e selecao de ordens do cluster para o ExecutionManager."""

from src.application.services.execution_direction import (
    build_execution_candidate,
    build_mandatory_fallback_candidate,
    mandatory_execution_eligible,
    recovery_execution_eligible,
)
from src.application.services.execution_symbols import (
    filter_execution_candidates,
    format_execution_alternates,
    select_best_execution_candidate,
    select_mandatory_execution_candidate,
)
from src.application.services.execution_symbols_recovery import (
    has_recovery_hedge_candidate,
    pending_recovery_active,
)
from src.domain.models.trade import TradeDirection


def apply_recovery_hedge_to_candidates(
    exec_mgr,
    candidates: list[tuple[str, TradeDirection, dict]],
    decisions: dict,
    *,
    cid: str,
    mandatory: bool = False,
) -> list[tuple[str, TradeDirection, dict]]:
    """Valida recovery na mesma direcao CALL/PUT do ultimo loss."""
    recovery_active = pending_recovery_active(getattr(exec_mgr.orch.risk_manager, "pending_loss", {}))
    if not recovery_active:
        return candidates
    last_loss = getattr(exec_mgr.orch.risk_manager, "last_loss_symbol", None)
    last_loss_dir = getattr(exec_mgr.orch.risk_manager, "last_loss_direction", None)
    if not has_recovery_hedge_candidate(
        candidates,
        last_loss_symbol=last_loss,
        last_loss_direction=last_loss_dir,
    ):
        if mandatory:
            fallback = build_mandatory_fallback_candidate(
                exec_mgr._trade_symbols(),
                decisions,
                recovery_active=True,
                last_loss_symbol=last_loss,
                last_loss_direction=last_loss_dir,
            )
            if fallback:
                return [fallback]
        exec_mgr.logger.warning(
            "[%s] RECOVERY_SKIP | sem candidato na mesma direcao do loss",
            cid,
        )
        return []
    return candidates


def _cluster_entry_eligible(
    entry: dict,
    *,
    mandatory: bool,
    recovery_active: bool,
    recovery_cfg: dict,
) -> bool:
    """Indica se entrada DL pode entrar no pool de candidatos do ciclo."""
    may_execute = bool(entry.get("metrics", {}).get("execute", False))
    if may_execute:
        return True
    if mandatory and mandatory_execution_eligible(entry):
        return True
    if recovery_active:
        return recovery_execution_eligible(entry, recovery_cfg)
    return False


def _mandatory_fallback_candidates(
    exec_mgr,
    decisions: dict,
    *,
    recovery_active: bool,
    last_loss_symbol: str | None,
    last_loss_direction: str | None,
) -> list[tuple[str, TradeDirection, dict]]:
    """Monta lista com candidato forcado quando o pool DL fica vazio."""
    fallback = build_mandatory_fallback_candidate(
        exec_mgr._trade_symbols(),
        decisions,
        recovery_active=recovery_active,
        last_loss_symbol=last_loss_symbol,
        last_loss_direction=last_loss_direction,
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
) -> list[tuple[str, TradeDirection, dict]]:
    """Coleta candidatos DL elegiveis para o ciclo atual."""
    candidates: list[tuple[str, TradeDirection, dict]] = []
    for symbol in exec_mgr._trade_symbols():
        entry = decisions.get(symbol)
        if not entry:
            continue
        if not _cluster_entry_eligible(
            entry,
            mandatory=mandatory,
            recovery_active=recovery_active,
            recovery_cfg=recovery_cfg,
        ):
            exec_mgr.logger.debug("[%s] SKIP: Conviccao insuficiente para %s (Metrics Gate)", cid, symbol)
            continue
        built = build_execution_candidate(symbol, entry)
        if built is None:
            continue
        candidates.append(built)
    return candidates


def collect_cluster_orders(exec_mgr, decisions: dict) -> list[tuple[str, TradeDirection, dict]]:
    """Seleciona uma ordem por ciclo; modo obrigatorio ignora gate execute=false."""
    mandatory = exec_mgr._mandatory_trade_each_cycle()
    recovery_active = pending_recovery_active(getattr(exec_mgr.orch.risk_manager, "pending_loss", {}))
    dl_cfg = exec_mgr.orch.config.get("deep_learning", {})
    recovery_cfg = dl_cfg.get("recovery_gating", {}) if isinstance(dl_cfg, dict) else {}
    cid = f"C{int(exec_mgr.orch._active_cycle_id):04d}"
    last_loss = getattr(exec_mgr.orch.risk_manager, "last_loss_symbol", None)
    last_loss_dir = getattr(exec_mgr.orch.risk_manager, "last_loss_direction", None)
    candidates = _gather_cluster_candidates(
        exec_mgr,
        decisions,
        mandatory=mandatory,
        recovery_active=recovery_active,
        recovery_cfg=recovery_cfg,
        cid=cid,
    )
    if not candidates and mandatory:
        candidates = _mandatory_fallback_candidates(
            exec_mgr,
            decisions,
            recovery_active=recovery_active,
            last_loss_symbol=last_loss,
            last_loss_direction=last_loss_dir,
        )
    if not candidates:
        return []
    candidates = apply_recovery_hedge_to_candidates(exec_mgr, candidates, decisions, cid=cid, mandatory=mandatory)
    if not candidates and mandatory:
        candidates = _mandatory_fallback_candidates(
            exec_mgr,
            decisions,
            recovery_active=recovery_active,
            last_loss_symbol=last_loss,
            last_loss_direction=last_loss_dir,
        )
    if not candidates:
        return []

    if not mandatory:
        dl_cfg = exec_mgr.orch.config.get("deep_learning", {})
        selection = dl_cfg.get("selection", {}) if isinstance(dl_cfg, dict) else {}
        filtered = filter_execution_candidates(candidates, selection=selection)
        if not filtered:
            return []
        candidates = filtered

    exec_cfg = exec_mgr.orch.config.get("orchestrator", {}).get("execution", {})
    margin = float(exec_cfg.get("diversify_after_loss_margin", 0.08))
    if mandatory:
        best = select_mandatory_execution_candidate(
            exec_mgr.orch,
            candidates,
            last_loss_symbol=last_loss,
            last_loss_direction=last_loss_dir,
            diversify_margin=margin,
            recovery_active=recovery_active,
        )
        if best is None:
            return []
    else:
        best = select_best_execution_candidate(
            candidates,
            last_loss_symbol=last_loss,
            last_loss_direction=last_loss_dir,
            diversify_margin=margin,
            recovery_active=recovery_active,
        )
    metrics = best[2]
    inv_tag = " inv" if metrics.get("direction_inverted") and not metrics.get("recovery_forced") else ""
    dl_name = metrics.get("dl_direction", best[1].name)
    exec_mgr.logger.info(
        "[%s] EXEC_SEL | %s ord=%s dl=%s%s s=%.2f v=%.2f r=%.2f | alt=%s",
        cid,
        best[0],
        best[1].name,
        dl_name,
        inv_tag,
        float(metrics.get("trade_score", metrics.get("conviction", 0.0))),
        float(metrics.get("val_accuracy", 0.0)),
        float(metrics.get("raw_prob", metrics.get("raw_conviction", 0.0))),
        format_execution_alternates(candidates, exclude_symbol=best[0]),
    )
    return [best]
