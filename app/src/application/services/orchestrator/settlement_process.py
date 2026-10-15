"""Aplicacao do outcome com resumo de cluster."""

from typing import Any

from src.application.services.orchestrator.settlement_cluster_log import log_cluster_summary
from src.application.services.orchestrator.settlement_outcome import process_contract_outcome


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
