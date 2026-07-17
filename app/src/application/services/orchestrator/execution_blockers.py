"""Logs de bloqueio de execucao quando nenhuma ordem e enviada."""

from src.application.services.log_dedupe import clear_log_channel, log_info_if_changed


def _candidate_block_reason(metrics: dict) -> str | None:
    """Extrai motivo de bloqueio sniper/quality do candidato."""
    reason = metrics.get("gate_reason") or metrics.get("quality_gate_reason")
    if isinstance(reason, str) and reason.strip():
        return reason.strip()
    if metrics.get("quality_guard_reject"):
        return "quality_guard_reject"
    if metrics.get("signal_status") == "SIGNAL_SUSPENDED":
        return "SIGNAL_SUSPENDED"
    return None


def log_execution_blockers(executor, decisions: dict, *, pending: float = 0.0) -> None:
    """Registra treino em andamento, rejeicoes sniper e recovery sem ordem."""
    cid = f"C{int(executor.orch._active_cycle_id):04d}"
    training: list[str] = []
    blocked: list[str] = []
    for symbol in executor._trade_symbols():
        entry = decisions.get(symbol)
        if not entry:
            continue
        metrics = entry.get("metrics") or {}
        if metrics.get("gate_reason") == "training":
            training.append(symbol)
            continue
        reason = _candidate_block_reason(metrics if isinstance(metrics, dict) else {})
        if reason:
            blocked.append(f"{symbol}:{reason}")
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
        log_info_if_changed(
            executor.orch,
            executor.logger,
            "exec_empty_blocks",
            "|".join(blocked),
            "[%s] EXEC_EMPTY || sem ordem | %s",
            cid,
            " ".join(blocked),
        )
    if float(pending) > 0.0:
        executor.logger.info(
            "[%s] EXEC_EMPTY || recovery sem ordem | pend=$%.2f | linear=%d",
            cid,
            float(pending),
            int(getattr(executor.orch.risk_manager, "consecutive_losses_linear", 0)),
        )
