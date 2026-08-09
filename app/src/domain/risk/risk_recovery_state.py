"""Políticas de estado de recuperação de risco e trava anti-tendência (AntiTrendLock)."""

from typing import Any

from src.domain.math.probability_entropy import binary_entropy
from src.domain.models.trade import TradeDirection
from src.domain.risk.recovery_state_config import load_recovery_state_from_settings
from src.domain.risk.soft_recovery_policy import load_soft_recovery_from_settings
from src.domain.symbols.drift_symbols import TRADING_SYMBOLS


DRIFT_PAIR_SYMBOLS = frozenset(TRADING_SYMBOLS)


def _rs() -> dict:
    """Resolve ou aplica  rs."""
    return load_recovery_state_from_settings()


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
    if not isinstance(soft, dict) or "dust_pending_clear_max" not in soft:
        soft = load_soft_recovery_from_settings()
    dust_max = float(soft["dust_pending_clear_max"])
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
        logger.debug(
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
        risk_manager.logger.debug(
            "RISK: Ciclo negativo (P&L: $%.2f) | pend=$%.2f | pnl_sess=$%+.2f | linear=%d",
            cluster_profit,
            pending,
            pnl_sess,
            risk_manager.consecutive_losses_linear,
        )
        return False
    if dust_cleared:
        risk_manager.logger.debug(
            "RISK: WIN operacional com dust clear (P&L: $%.2f) | pnl_sess=$%+.2f | regime=EXPLORE",
            cluster_profit,
            pnl_sess,
        )
        risk_manager._linear_reset_occurred = True
        return True
    if pending > 0.0:
        apply_dlambert_partial_win_retraction(risk_manager)
        risk_manager.logger.debug(
            "RISK: WIN operacional (P&L: $%.2f) | pend=$%.2f | pnl_sess=$%+.2f | linear=%d",
            cluster_profit,
            pending,
            pnl_sess,
            risk_manager.consecutive_losses_linear,
        )
        return False
    linear_reset = linear_before > 0 or linear > 0
    if linear_reset:
        risk_manager.logger.debug(
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
        risk_manager.logger.debug(
            "RISK: Lucro parcial $%.2f | pend=$%.2f | pnl_sess=$%+.2f | linear=%d",
            profit,
            pending_after,
            float(risk_manager.total_session_profit),
            int(risk_manager.consecutive_losses_linear),
        )
    return pending_after


def critical_recovery_stress(linear_losses: int, pending_total: float) -> bool:
    """Resolve ou aplica critical recovery stress."""
    rs = _rs()
    return int(linear_losses) >= int(rs["critical_linear_losses"]) and float(pending_total) > float(
        rs["critical_pending_total"]
    )


def tcn_macro_ultra_extreme_conviction(raw_prob: float, direction: str) -> bool:
    """Resolve ou aplica tcn macro ultra extreme conviction."""
    rs = _rs()
    prob = float(raw_prob)
    side = str(direction or "").upper()
    if side == TradeDirection.PUT.name:
        return prob <= float(rs["put_extreme_raw_prob"])
    if side == TradeDirection.CALL.name:
        return prob >= float(rs["call_extreme_raw_prob"])
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
    rs = _rs()
    raw_prob = float(metrics["raw_prob"]) if "raw_prob" in metrics else float(rs["raw_prob_default"])
    return tcn_macro_ultra_extreme_conviction(raw_prob, direction)


def cointegration_redirect_armed(
    initial_bankroll: float,
    pending_total: float,
    *,
    threshold: float | None = None,
) -> bool:
    """Resolve ou aplica cointegration redirect armed."""
    rs = _rs()
    bankroll = float(initial_bankroll)
    if bankroll <= 0.0 or bankroll > float(rs["micro_bankroll_threshold"]):
        return False
    limit = float(threshold) if threshold is not None else float(rs["cointegration_drawdown_fraction"]) * bankroll
    return float(pending_total) > limit


def micro_tail_stake_cap(bankroll: float) -> float:
    """Resolve ou aplica micro tail stake cap."""
    rs = _rs()
    bal = float(bankroll)
    pct = (
        float(rs["micro_unit_floor"])
        if bal <= float(rs["micro_bankroll_threshold"])
        else float(rs["micro_unit_bankroll_pct"])
    )
    unit = max(0.0, bal) * pct
    return float(rs["micro_tail_unit_multiplier"]) * unit


def cointegration_pair_score(metrics: dict[str, Any]) -> float:
    """Resolve ou aplica cointegration pair score."""
    rs = _rs()
    default_prob = float(rs["raw_prob_default"])
    if "calibrated_prob" in metrics:
        prob = float(metrics["calibrated_prob"])
    elif "raw_prob" in metrics:
        prob = float(metrics["raw_prob"])
    else:
        prob = default_prob
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
