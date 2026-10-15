"""Aplicacao do outcome com resumo de cluster e log contextual de liquidacao."""

from typing import Any

from src.application.services.orchestrator.settlement_cluster_log import log_cluster_summary
from src.application.services.orchestrator.settlement_outcome import process_contract_outcome
from src.presentation.terminal.log_context import (
    bind_log_context,
    format_log_context_prefix,
    get_log_context,
    restore_log_context,
)


def process_settlement_outcome(
    orch: Any,
    c: dict,
    contract: Any,
    c_id: int,
    profit: float,
    *,
    audit_direction: str | None = None,
    audit_raw_prob: float | None = None,
) -> None:
    """Atualiza saldo, risco e estatisticas de uma liquidacao."""
    process_contract_outcome(
        orch,
        c,
        contract,
        c_id,
        profit,
        audit_direction=audit_direction,
        audit_raw_prob=audit_raw_prob,
        log_cluster_summary=log_cluster_summary,
    )


def emit_contract_settlement_log(orch: Any, line: str, *, c_id: int, sym: str | None = None) -> None:
    """Emite linha [RESOLVED] com o contexto exato do ciclo e ativo do contrato."""
    orig_cycle = getattr(orch, "_contract_cycle", {}).get(c_id, None)
    symbol = sym or getattr(getattr(orch, "risk_manager", None), "contract_to_symbol", {}).get(c_id)
    prev_cid, prev_sym = get_log_context()
    bind_log_context(cycle_id=orig_cycle, symbol=symbol)
    try:
        if getattr(orch, "_buffer_result_logs", False):
            prefix = format_log_context_prefix()
            target_line = f"{prefix}{line}" if prefix and not line.startswith("[c") else line
            orch._pending_result_logs.append(target_line)
        else:
            orch.logger.info(line)
    finally:
        restore_log_context(prev_cid, prev_sym)
