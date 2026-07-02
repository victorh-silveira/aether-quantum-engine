"""Estado financeiro de recovery: pending_loss e perdas lineares D'Alembert."""


def pending_loss_total(pending_loss: dict[str, float]) -> float:
    """Soma perdas pendentes da sessao."""
    return sum(float(v) for v in pending_loss.values())


def recovery_financially_active(pending_loss: dict[str, float]) -> bool:
    """True enquanto houver drawdown financeiro pendente na sessao."""
    return pending_loss_total(pending_loss) > 0.0


def apply_win_to_pending_loss(pending_loss: dict[str, float], profit: float) -> None:
    """Reduz perdas pendentes com lucro parcial de um contrato."""
    remaining_profit = profit
    for sym in list(pending_loss.keys()):
        if remaining_profit <= 0:
            break
        current_loss = pending_loss[sym]
        if current_loss <= remaining_profit:
            remaining_profit -= current_loss
            pending_loss[sym] = 0.0
        else:
            pending_loss[sym] = current_loss - remaining_profit
            remaining_profit = 0.0


def apply_dlambert_partial_win_retraction(risk_manager) -> None:
    """Retrai contador linear em 1 unidade apos WIN parcial em recovery."""
    linear = int(getattr(risk_manager, "consecutive_losses_linear", 0))
    if linear <= 0:
        return
    risk_manager.consecutive_losses_linear = max(1, linear - 1)


def apply_cluster_profit_to_recovery_state(risk_manager, cluster_profit: float) -> None:
    """Atualiza perdas lineares sem reset falso enquanto pending_loss > 0."""
    pending = pending_loss_total(risk_manager.pending_loss)
    pnl_sess = float(risk_manager.total_session_profit)
    linear = int(getattr(risk_manager, "consecutive_losses_linear", 0))
    if cluster_profit < 0.0:
        risk_manager.consecutive_losses_linear = linear + 1
        risk_manager.logger.info(
            "RISK: Ciclo negativo (P&L: $%.2f) | pend=$%.2f | pnl_sess=$%+.2f | linear=%d",
            cluster_profit,
            pending,
            pnl_sess,
            risk_manager.consecutive_losses_linear,
        )
        return
    if pending > 0.0:
        apply_dlambert_partial_win_retraction(risk_manager)
        risk_manager.logger.info(
            "RISK: WIN operacional (P&L: $%.2f) | pend=$%.2f | pnl_sess=$%+.2f | linear=%d",
            cluster_profit,
            pending,
            pnl_sess,
            risk_manager.consecutive_losses_linear,
        )
        return
    if linear > 0:
        risk_manager.logger.info(
            "RISK: Recovery financeiro zerado (P&L: $%.2f) | pnl_sess=$%+.2f | reset linear",
            cluster_profit,
            pnl_sess,
        )
    risk_manager.consecutive_losses_linear = 0
    risk_manager.last_loss_stake = 0.0


def log_partial_win_recovery(risk_manager, profit: float) -> float:
    """Registra lucro parcial que ainda nao extingue o pending_loss da sessao."""
    pending_after = pending_loss_total(risk_manager.pending_loss)
    if pending_after > 0.0:
        risk_manager.logger.info(
            "RISK: Lucro parcial $%.2f | pend=$%.2f | pnl_sess=$%+.2f | linear=%d",
            profit,
            pending_after,
            float(risk_manager.total_session_profit),
            int(risk_manager.consecutive_losses_linear),
        )
    return pending_after
