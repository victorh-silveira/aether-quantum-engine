"""Políticas de estado de recuperação de risco e trava anti-tendência (AntiTrendLock)."""

from typing import Any

from src.domain.math.probability_entropy import binary_entropy
from src.domain.models.trade import TradeDirection
from src.domain.risk.soft_recovery_policy import DEFAULT_DUST_PENDING_CLEAR_MAX
from src.domain.symbols.drift_symbols import TRADING_SYMBOLS


CRITICAL_LINEAR_LOSSES = 5
CRITICAL_PENDING_TOTAL = 250.0
PUT_EXTREME_RAW_PROB = 0.18
CALL_EXTREME_RAW_PROB = 0.82
COINTEGRATION_DRAWDOWN_FRACTION = 0.15
MICRO_BANKROLL_THRESHOLD = 250.0
MICRO_TAIL_LINEAR_LEVEL = 4
MICRO_TAIL_UNIT_MULTIPLIER = 4.2
DRIFT_PAIR_SYMBOLS = frozenset(TRADING_SYMBOLS)


def pending_loss_total(pending_loss: dict[str, float]) -> float:
    """Soma perdas pendentes da sessao."""
    return sum(float(v) for v in pending_loss.values())


def recovery_financially_active(pending_loss: dict[str, float]) -> bool:
    """True enquanto houver drawdown financeiro pendente na sessao."""
    return pending_loss_total(pending_loss) > 0.0


def clear_dust_pending_loss(risk_manager: Any, *, soft_recovery: dict[str, Any] | None = None) -> bool:
    """Zera dust de pending e retorna sessao ao regime EXPLORE."""
    soft = soft_recovery
    if not isinstance(soft, dict):
        soft = getattr(risk_manager, "soft_recovery_config", None)
    if not isinstance(soft, dict):
        soft = {}
    dust_max = float(soft.get("dust_pending_clear_max", DEFAULT_DUST_PENDING_CLEAR_MAX))
    pending = getattr(risk_manager, "pending_loss", None)
    if not isinstance(pending, dict):
        return False
    total = pending_loss_total(pending)
    if total <= 0.0 or total > dust_max:
        return False
    pending.clear()
    risk_manager.consecutive_losses_linear = 0
    if hasattr(risk_manager, "last_loss_stake"):
        risk_manager.last_loss_stake = 0.0
    logger = getattr(risk_manager, "logger", None)
    if logger is not None:
        logger.info(
            "RISK: Dust pending cleared | was=$%.2f | max=$%.2f | regime=EXPLORE",
            total,
            dust_max,
        )
    return True


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
    linear_before = int(getattr(risk_manager, "consecutive_losses_linear", 0))
    dust_cleared = clear_dust_pending_loss(risk_manager)
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
    if dust_cleared:
        risk_manager.logger.info(
            "RISK: WIN operacional com dust clear (P&L: $%.2f) | pnl_sess=$%+.2f | regime=EXPLORE",
            cluster_profit,
            pnl_sess,
        )
        risk_manager._linear_reset_occurred = True
        return True
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
    linear_reset = linear_before > 0 or linear > 0
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
    vol_ratio: float = 0.0,
    bb_width_zscore: float = 0.0,
) -> tuple[TradeDirection | None, str]:
    """Política pura de domínio para resolver a direção sob AntiTrendLock.

    Recebe inputs limpos e retorna a direção pura resolvida (ou None se suspensa)
    e a ação correspondente (ex: 'FLIP to PUT', 'FLIP to CALL', 'FREEZE: SKIP CYCLE', 'KEEP').
    """
    _ = (vol_ratio, bb_width_zscore)

    if consecutive_losses < 2:
        return proposed_direction, "KEEP"

    resolved: TradeDirection | None = None
    action: str = "FREEZE: SKIP CYCLE"

    is_bull = symbol == "RDBULL"
    is_bear = symbol == "RDBEAR"

    if is_bull and proposed_direction == TradeDirection.CALL:
        expanding = (bear_put_prob + 1e-12 > bull_call_prob) or (
            probability_delta > cross_symbol_prob_delta_mean + 1e-12
        )
        if expanding and predicted_payoff_edge + 1e-12 >= 0.0:
            resolved, action = TradeDirection.PUT, "FLIP to PUT"

    elif is_bull and proposed_direction == TradeDirection.PUT or is_bear and proposed_direction == TradeDirection.PUT:
        expanding = (bull_call_prob + 1e-12 > bear_put_prob) or (
            probability_delta > cross_symbol_prob_delta_mean + 1e-12
        )
        if expanding and predicted_payoff_edge + 1e-12 >= 0.0:
            resolved, action = TradeDirection.CALL, "FLIP to CALL"

    elif is_bear and proposed_direction == TradeDirection.CALL:
        expanding = (bear_put_prob + 1e-12 > bull_call_prob) or (
            probability_delta > cross_symbol_prob_delta_mean + 1e-12
        )
        if expanding and predicted_payoff_edge + 1e-12 >= 0.0:
            resolved, action = TradeDirection.PUT, "FLIP to PUT"

    return resolved, action


def critical_recovery_stress(linear_losses: int, pending_total: float) -> bool:
    """True quando streak e passivo exigem waiver de emergencia em conjunto."""
    return int(linear_losses) >= CRITICAL_LINEAR_LOSSES and float(pending_total) > CRITICAL_PENDING_TOTAL


def tcn_macro_ultra_extreme_conviction(raw_prob: float, direction: str) -> bool:
    """True quando a TCN macro exibe cauda extrema alinhada a PUT ou CALL."""
    prob = float(raw_prob)
    side = str(direction or "").upper()
    if side == TradeDirection.PUT.name:
        return prob <= PUT_EXTREME_RAW_PROB
    if side == TradeDirection.CALL.name:
        return prob >= CALL_EXTREME_RAW_PROB
    return False


def meta_payoff_veto_emergency_waiver(
    metrics: dict[str, Any],
    *,
    direction: str,
    risk_manager: Any | None = None,
) -> bool:
    """Libera o veto em situacao de estresse critico de recovery se a conviccao for extrema."""
    if risk_manager is None:
        return False
    linear = int(getattr(risk_manager, "consecutive_losses_linear", 0))
    if hasattr(risk_manager, "pending_loss_total") and callable(risk_manager.pending_loss_total):
        pending = float(risk_manager.pending_loss_total())
    else:
        pending = pending_loss_total(getattr(risk_manager, "pending_loss", {}))
    if not critical_recovery_stress(linear, pending):
        return False
    raw_prob = float(metrics.get("raw_prob", 0.5))
    return tcn_macro_ultra_extreme_conviction(raw_prob, direction)


def cointegration_redirect_armed(
    initial_bankroll: float,
    pending_total: float,
    *,
    threshold: float | None = None,
) -> bool:
    """True quando drawdown excede limiar configuravel (padrao 15% do capital vivo)."""
    bankroll = float(initial_bankroll)
    if bankroll <= 0.0 or bankroll > MICRO_BANKROLL_THRESHOLD:
        return False
    limit = float(threshold) if threshold is not None else COINTEGRATION_DRAWDOWN_FRACTION * bankroll
    return float(pending_total) > limit


def micro_tail_stake_cap(bankroll: float) -> float:
    """Teto de cauda 4.2*U para progressao soft a partir do nivel linear 4."""
    unit = max(0.0, float(bankroll)) * (0.01 if float(bankroll) <= MICRO_BANKROLL_THRESHOLD else 0.0015)
    return MICRO_TAIL_UNIT_MULTIPLIER * unit


def cointegration_pair_score(metrics: dict[str, Any]) -> float:
    """Pontua Drift por Z positivo maximizado e entropia de Shannon minimizada."""
    prob = float(metrics.get("calibrated_prob", metrics.get("raw_prob", 0.5)))
    z = float(metrics.get("meta_payoff_edge_zscore", metrics.get("edge_zscore", 0.0)))
    if z <= 0.0:
        return float("-inf")
    return float(z) - binary_entropy(prob)


def select_cointegration_redirect_candidate(
    candidates: list[tuple[str, Any, dict]],
) -> list[tuple[str, Any, dict]]:
    """Redireciona soft recovery ao Drift de menor entropia e maior Z positivo."""
    drift = [item for item in candidates if str(item[0]) in DRIFT_PAIR_SYMBOLS]
    if not drift:
        return []
    if len(drift) == 1:
        return drift
    return [max(drift, key=lambda item: cointegration_pair_score(item[2]))]
