"""Estado financeiro de recovery: pending_loss e perdas consecutivas."""


def pending_loss_total(pending_loss: dict[str, float]) -> float:
    """Soma perdas pendentes da sessao."""
    return sum(float(v) for v in pending_loss.values())


def recovery_financially_active(pending_loss: dict[str, float]) -> bool:
    """True enquanto houver drawdown financeiro pendente na sessao."""
    return pending_loss_total(pending_loss) > 0.0


def apply_cluster_profit_to_recovery_state(risk_manager, cluster_profit: float) -> None:
    """Atualiza perdas consecutivas sem reset falso enquanto pending_loss > 0."""
    pending = pending_loss_total(risk_manager.pending_loss)
    pnl_sess = float(risk_manager.total_session_profit)
    if cluster_profit < 0.0:
        risk_manager.consecutive_losses += 1
        risk_manager.logger.info(
            "RISK: Ciclo negativo (P&L: $%.2f) | pend=$%.2f | pnl_sess=$%+.2f | consecutive_losses=%d",
            cluster_profit,
            pending,
            pnl_sess,
            risk_manager.consecutive_losses,
        )
        return
    if pending > 0.0:
        risk_manager.logger.info(
            "RISK: WIN operacional (P&L: $%.2f) | pend=$%.2f | pnl_sess=$%+.2f | "
            "RECOVERY mantido consecutive_losses=%d",
            cluster_profit,
            pending,
            pnl_sess,
            risk_manager.consecutive_losses,
        )
        return
    if risk_manager.consecutive_losses > 0:
        risk_manager.logger.info(
            "RISK: Recovery financeiro zerado (P&L: $%.2f) | pnl_sess=$%+.2f | reset perdas consecutivas",
            cluster_profit,
            pnl_sess,
        )
    risk_manager.consecutive_losses = 0
    risk_manager.last_martingale_stake = 0.0
    risk_manager.last_loss_stake = 0.0


def log_partial_win_recovery(risk_manager, profit: float) -> float:
    """Registra lucro parcial que ainda nao extingue o pending_loss da sessao."""
    pending_after = pending_loss_total(risk_manager.pending_loss)
    if pending_after > 0.0:
        risk_manager.logger.info(
            "RISK: Lucro parcial $%.2f | pend=$%.2f | pnl_sess=$%+.2f | RECOVERY mantido losses=%d",
            profit,
            pending_after,
            float(risk_manager.total_session_profit),
            int(risk_manager.consecutive_losses),
        )
    return pending_after
