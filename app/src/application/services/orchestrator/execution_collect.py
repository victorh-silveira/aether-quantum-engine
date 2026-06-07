"""Montagem e selecao de ordens do cluster para o ExecutionManager."""

from src.application.services.execution_direction import build_execution_candidate
from src.application.services.execution_symbols import (
    filter_execution_candidates,
    format_execution_alternates,
    select_best_execution_candidate,
    select_mandatory_execution_candidate,
)
from src.application.services.execution_symbols_recovery import (
    has_recovery_hedge_candidate,
    inject_recovery_hedge_candidates,
    pending_recovery_active,
)
from src.domain.models.trade import TradeDirection


def apply_recovery_hedge_to_candidates(
    exec_mgr,
    candidates: list[tuple[str, TradeDirection, dict]],
    decisions: dict,
    *,
    cid: str,
) -> list[tuple[str, TradeDirection, dict]]:
    """Injeta hedge e retorna lista vazia quando recovery exige hedge indisponivel."""
    recovery_active = pending_recovery_active(getattr(exec_mgr.orch.risk_manager, "pending_loss", {}))
    if not recovery_active:
        return candidates
    last_loss = getattr(exec_mgr.orch.risk_manager, "last_loss_symbol", None)
    last_loss_dir = getattr(exec_mgr.orch.risk_manager, "last_loss_direction", None)
    candidates = inject_recovery_hedge_candidates(
        candidates,
        decisions,
        last_loss_symbol=last_loss,
        last_loss_direction=last_loss_dir,
    )
    exec_cfg = exec_mgr.orch.config.get("orchestrator", {}).get("execution", {})
    require_hedge = bool(exec_cfg.get("recovery_require_hedge", True))
    if require_hedge and not has_recovery_hedge_candidate(
        candidates,
        last_loss_symbol=last_loss,
        last_loss_direction=last_loss_dir,
    ):
        exec_mgr.logger.warning(
            "[%s] RECOVERY_SKIP | sem candidato de hedge no par apos loss",
            cid,
        )
        return []
    return candidates


def collect_cluster_orders(exec_mgr, decisions: dict) -> list[tuple[str, TradeDirection, dict]]:
    """Seleciona uma ordem por ciclo; modo obrigatorio ignora gate execute=false."""
    mandatory, invert = exec_mgr._execution_flags()
    recovery_active = pending_recovery_active(getattr(exec_mgr.orch.risk_manager, "pending_loss", {}))
    candidates: list[tuple[str, TradeDirection, dict]] = []
    cid = f"C{int(exec_mgr.orch._active_cycle_id):04d}"
    for symbol in exec_mgr._trade_symbols():
        entry = decisions.get(symbol)
        if not entry:
            continue
        if (recovery_active or not mandatory) and not entry.get("metrics", {}).get("execute", True):
            exec_mgr.logger.debug("[%s] SKIP: Conviccao insuficiente para %s (Metrics Gate)", cid, symbol)
            continue
        built = build_execution_candidate(symbol, entry, invert_dl_direction=invert)
        if built is None:
            continue
        candidates.append(built)

    if not candidates:
        return []

    candidates = apply_recovery_hedge_to_candidates(exec_mgr, candidates, decisions, cid=cid)
    if not candidates:
        return []

    recovery_active = pending_recovery_active(getattr(exec_mgr.orch.risk_manager, "pending_loss", {}))
    last_loss = getattr(exec_mgr.orch.risk_manager, "last_loss_symbol", None)
    last_loss_dir = getattr(exec_mgr.orch.risk_manager, "last_loss_direction", None)

    if not mandatory:
        dl_cfg = exec_mgr.orch.config.get("deep_learning", {})
        selection = dl_cfg.get("selection", {}) if isinstance(dl_cfg, dict) else {}
        filtered = filter_execution_candidates(candidates, selection=selection)
        if not filtered:
            return []
        candidates = filtered

    exec_cfg = exec_mgr.orch.config.get("orchestrator", {}).get("execution", {})
    margin = float(exec_cfg.get("diversify_after_loss_margin", 0.08))
    dl_cfg = exec_mgr.orch.config.get("deep_learning", {})
    flip_raw_min = float(dl_cfg.get("post_loss_flip_raw_min", 0.58)) if isinstance(dl_cfg, dict) else 0.58
    if mandatory:
        best = select_mandatory_execution_candidate(
            exec_mgr.orch,
            candidates,
            last_loss_symbol=last_loss,
            last_loss_direction=last_loss_dir,
            diversify_margin=margin,
            recovery_active=recovery_active,
            flip_raw_min=flip_raw_min,
        )
    else:
        best = select_best_execution_candidate(
            candidates,
            last_loss_symbol=last_loss,
            last_loss_direction=last_loss_dir,
            diversify_margin=margin,
            recovery_active=recovery_active,
        )
    metrics = best[2]
    inv_tag = " inv" if metrics.get("direction_inverted") else ""
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
