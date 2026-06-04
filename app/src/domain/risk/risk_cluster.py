"""Finalizacao de cluster e estatisticas de perdas consecutivas."""


def finalize_risk_cluster(risk_manager) -> None:
    """Atualiza perdas consecutivas e limpa estado do cluster."""
    cluster_profit = sum(risk_manager.cluster_results.values())
    if cluster_profit < 0.0:
        risk_manager.consecutive_losses += 1
        risk_manager.logger.info(
            "RISK: Ciclo negativo (P&L: $%.2f) consecutive_losses=%d",
            cluster_profit,
            risk_manager.consecutive_losses,
        )
    else:
        if risk_manager.consecutive_losses > 0:
            risk_manager.logger.info(
                "RISK: Ciclo positivo (P&L: $%.2f). Reset perdas consecutivas",
                cluster_profit,
            )
        risk_manager.consecutive_losses = 0
        if sum(risk_manager.pending_loss.values()) <= 0.0:
            risk_manager.last_martingale_stake = 0.0
            risk_manager.last_loss_stake = 0.0

    risk_manager._cooldown_until_mono = 0.0
    risk_manager.current_cooldown_ticks = 0
    risk_manager.active_contract_ids = []
    risk_manager.contract_to_symbol = {}
    risk_manager.cluster_results = {}
    risk_manager.expected_cluster_settlements = 0
