"""Finalizacao de cluster e estatisticas de perdas consecutivas."""

from src.domain.risk.risk_recovery_state import apply_cluster_profit_to_recovery_state


def finalize_risk_cluster(risk_manager) -> None:
    """Atualiza perdas consecutivas e limpa estado do cluster."""
    cluster_profit = sum(risk_manager.cluster_results.values())
    apply_cluster_profit_to_recovery_state(risk_manager, cluster_profit)

    risk_manager._cooldown_until_mono = 0.0
    risk_manager.current_cooldown_ticks = 0
    risk_manager.active_contract_ids = []
    risk_manager.contract_to_symbol = {}
    risk_manager.cluster_results = {}
    risk_manager.expected_cluster_settlements = 0
