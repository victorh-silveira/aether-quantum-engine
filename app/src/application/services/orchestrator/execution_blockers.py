"""Logs de bloqueio de execucao quando nenhuma ordem e enviada."""

from src.application.services.execution_direction import infer_dl_direction
from src.application.services.log_dedupe import clear_log_channel, log_info_if_changed


def _hold_token(symbol: str, entry: dict) -> str | None:
    """Resume simbolo bloqueado com direcao DL mas sem execute=true."""
    metrics = entry.get("metrics") or {}
    if metrics.get("gate_reason") == "training" or metrics.get("execute"):
        return None
    direction = infer_dl_direction(entry)
    if direction is None:
        return None
    gate = str(metrics.get("gate_reason") or "block")
    raw = metrics.get("raw_prob")
    if raw is not None:
        return f"{symbol}:{direction.name}:{gate}:r{float(raw):.2f}"
    return f"{symbol}:{direction.name}:{gate}"


def log_execution_blockers(executor, decisions: dict) -> None:
    """Registra treino em andamento e sinais DL que nao viraram ordem."""
    cid = f"C{int(executor.orch._active_cycle_id):04d}"
    training: list[str] = []
    holds: list[str] = []
    for symbol in executor._trade_symbols():
        entry = decisions.get(symbol)
        if not entry:
            continue
        metrics = entry.get("metrics") or {}
        if metrics.get("gate_reason") == "training":
            training.append(symbol)
            continue
        token = _hold_token(symbol, entry)
        if token:
            holds.append(token)
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
    if holds:
        log_info_if_changed(
            executor.orch,
            executor.logger,
            "exec_hold",
            "|".join(holds),
            "[%s] EXEC_HOLD | %s",
            cid,
            ",".join(holds[:3]),
        )
    elif clear_log_channel(executor.orch, "exec_hold"):
        executor.logger.debug("[%s] EXEC_HOLD || liberado", cid)
