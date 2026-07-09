"""Políticas de estado de recuperação de risco e trava anti-tendência (AntiTrendLock)."""

from src.domain.models.trade import TradeDirection


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


def apply_cluster_profit_to_recovery_state(risk_manager, cluster_profit: float) -> bool:
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
        return False
    if pending > 0.0:
        apply_dlambert_partial_win_retraction(risk_manager)
        risk_manager.logger.info(
            "RISK: WIN operacional (P&L: $%.2f) | pend=$%.2f | pnl_sess=$%+.2f | linear=%d",
            cluster_profit,
            pending,
            pnl_sess,
            risk_manager.consecutive_losses_linear,
        )
        return False
    linear_reset = linear > 0
    if linear_reset:
        risk_manager.logger.info(
            "RISK: Recovery financeiro zerado (P&L: $%.2f) | pnl_sess=$%+.2f | reset linear",
            cluster_profit,
            pnl_sess,
        )
    risk_manager.consecutive_losses_linear = 0
    risk_manager.last_loss_stake = 0.0
    if linear_reset:
        risk_manager._linear_reset_occurred = True
    return linear_reset


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


def evaluate_anti_trend_lock(
    symbol: str,
    proposed_direction: TradeDirection,
    consecutive_losses: int,
    bull_call_prob: float,
    bear_put_prob: float,
    probability_delta: float,
    predicted_payoff_edge: float,
    cross_symbol_prob_delta_mean: float,
) -> tuple[TradeDirection | None, str]:
    """Política pura de domínio para resolver a direção sob AntiTrendLock.

    Recebe inputs limpos e retorna a direção pura resolvida (ou None se suspensa)
    e a ação correspondente (ex: 'FLIP to PUT', 'FLIP to CALL', 'FREEZE: SKIP CYCLE', 'KEEP').
    """
    if consecutive_losses < 2:
        return proposed_direction, "KEEP"

    if symbol == "RDBULL" and proposed_direction == TradeDirection.CALL:
        # Tentamos fazer flip para PUT (que opera em RDBEAR)
        expanding = (bear_put_prob + 1e-12 > bull_call_prob) or (
            probability_delta > cross_symbol_prob_delta_mean + 1e-12
        )
        if expanding and predicted_payoff_edge + 1e-12 >= 0.0:
            return TradeDirection.PUT, "FLIP to PUT"
        return None, "FREEZE: SKIP CYCLE"

    if symbol == "RDBEAR" and proposed_direction == TradeDirection.PUT:
        # Tentamos fazer flip para CALL (que opera em RDBULL)
        expanding = (bull_call_prob + 1e-12 > bear_put_prob) or (
            probability_delta > cross_symbol_prob_delta_mean + 1e-12
        )
        if expanding and predicted_payoff_edge + 1e-12 >= 0.0:
            return TradeDirection.CALL, "FLIP to CALL"
        return None, "FREEZE: SKIP CYCLE"

    return None, "FREEZE: SKIP CYCLE"
