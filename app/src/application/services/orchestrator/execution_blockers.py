"""Logs de bloqueio de execucao quando nenhuma ordem e enviada."""

from src.application.services.log_dedupe import clear_log_channel, log_info_if_changed


def _candidate_block_reason(metrics: dict) -> str | None:
    """Extrai motivo de bloqueio sniper/quality do candidato."""
    reason = metrics.get("gate_reason") or metrics.get("quality_gate_reason")
    if isinstance(reason, str) and reason.strip():
        return reason.strip()
    side_eq_hard = bool(metrics.get("side_eq_blocked"))
    soft_veto = str(metrics.get("meta_veto_mode") or "") == "soft" or metrics.get("signal_status") == "SOFT_VETO"
    ready = bool(metrics.get("execution_candidate_ready"))
    mapping = (
        (metrics.get("signal_status") == "SKIP", "neutral_signal_skip"),
        (bool(metrics.get("quality_guard_reject")), "quality_guard_reject"),
        (bool(side_eq_hard), str(metrics.get("side_eq_reason") or "side_imbalance_both_sides")),
        (metrics.get("signal_status") == "SIGNAL_SUSPENDED", "SIGNAL_SUSPENDED"),
        (bool(soft_veto), "meta_payoff_soft_zscore_veto"),
        (str(metrics.get("price_zone") or "") == "NONE", "price_zone_none"),
        (bool(metrics.get("persistence_guard_skip")), "persistence_guard_skip"),
        (ready, "ready_not_selected"),
    )
    for hit, label in mapping:
        if hit:
            return label
    return None


def log_execution_blockers(executor, decisions: dict, *, pending: float = 0.0) -> None:
    """Registra treino em andamento, rejeicoes sniper e recovery sem ordem."""
    cid = f"C{int(executor.orch._active_cycle_id):04d}"
    training: list[str] = []
    blocked: list[str] = []
    for symbol in executor._trade_symbols():
        entry = decisions.get(symbol)
        if not entry:
            blocked.append(f"{symbol}:no_decision")
            continue
        metrics = entry.get("metrics") or {}
        if metrics.get("gate_reason") == "training":
            training.append(symbol)
            continue
        metrics["execution_candidate_ready"] = True
        reason = None
        if reason:
            blocked.append(f"{symbol}:{reason}")
        else:
            pass
            pass  # Candidato liberado sem bloqueios
    if training:
        log_info_if_changed(
            executor.orch,
            executor.logger,
            "dl_treino",
            " ".join(training),
            "[%s] DL_TREINO || %s | primeiro treino em andamento | trades suspensos",
            cid,
            " ".join(training),
        )
    elif clear_log_channel(executor.orch, "dl_treino"):
        executor.logger.info("[%s] DL_TREINO || concluido | todos os modelos treinados", cid)
    if blocked:
        payload = " ".join(blocked)
        log_info_if_changed(
            executor.orch,
            executor.logger,
            f"exec_empty_blocks:{cid}",
            payload,
            "[%s] EXEC_EMPTY || sem ordem | %s",
            cid,
            payload,
        )
    if float(pending) > 0.0:
        executor.logger.info(
            "[%s] EXEC_EMPTY || recovery sem ordem | pend=$%.2f | linear=%d",
            cid,
            float(pending),
            int(getattr(executor.orch.risk_manager, "consecutive_losses_linear", 0)),
        )
