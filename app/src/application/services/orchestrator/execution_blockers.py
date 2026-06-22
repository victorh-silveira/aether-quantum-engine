"""Logs de bloqueio de execucao quando nenhuma ordem e enviada."""

from src.application.services.log_dedupe import clear_log_channel, log_info_if_changed


def log_execution_blockers(executor, decisions: dict) -> None:
    """Registra treino em andamento."""
    cid = f"C{int(executor.orch._active_cycle_id):04d}"
    training: list[str] = []
    for symbol in executor._trade_symbols():
        entry = decisions.get(symbol)
        if not entry:
            continue
        metrics = entry.get("metrics") or {}
        if metrics.get("gate_reason") == "training":
            training.append(symbol)
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
