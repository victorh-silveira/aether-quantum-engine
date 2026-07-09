"""Fluxo auxiliar de stake para RiskManager."""

from typing import Any

from src.domain.risk.stake_sizing import compute_single_strike_kelly_base
from src.domain.risk.stake_target_proximity import apply_target_proximity_damping
from src.domain.risk.stop_win_target import persisted_session_target, resolve_stop_win_target


def apply_stop_win_kelly_boost(
    rm: Any,
    *,
    kelly_base: float,
    bankroll: float,
    payout: float,
    sizing_conviction: float,
    conviction: float,
    dl_execute: bool,
    recovery_active: bool,
    apply_stop_win: bool,
    silent: bool,
) -> float:
    """Aplica boost de stake Kelly alinhado ao stop win diario."""
    min_stop_conv = float(rm.kelly_config.get("stop_win_kelly_min_conviction", 0.45))
    stop_win_boost_ok = dl_execute or sizing_conviction + 1e-9 >= min_stop_conv
    if not apply_stop_win or recovery_active or not stop_win_boost_ok:
        return kelly_base
    boosted = compute_single_strike_kelly_base(
        kelly_base,
        bankroll,
        payout,
        sizing_conviction if not dl_execute else conviction,
        rm.config,
        rm.kelly_config,
        rm.initial_bankroll,
        rm.total_session_profit,
        has_active_contracts=bool(rm.active_contract_ids),
    )
    if boosted > kelly_base and not silent:
        rm.logger.info(
            "RISK: STOP WIN KELLY | stake $%.2f -> $%.2f (meta sessao)",
            kelly_base,
            boosted,
        )
    return boosted


def apply_target_proximity_to_kelly(rm: Any, kelly_base: float, *, apply_stop_win: bool) -> float:
    """Comprime Kelly bruto pela proximidade da meta de stop win da sessao."""
    if not apply_stop_win:
        return kelly_base
    target = resolve_stop_win_target(
        rm.config,
        rm.initial_bankroll,
        persisted_target=persisted_session_target(rm),
    )
    return apply_target_proximity_damping(kelly_base, target, rm.total_session_profit)


def emit_cycle_stake_log(
    rm: Any,
    *,
    cycle_id: int,
    silent: bool,
    mode_tag: str,
    final_stake: float,
    f_star: float,
    p: float,
    b: float,
    bankroll: float,
    loss_to_recover: float,
    linear_losses: int,
    symbol: str,
    rec_info: str,
) -> None:
    """Emite log estruturado de stake do ciclo quando cycle_id e visivel."""
    if cycle_id <= 0 or silent:
        return
    rm.logger.info(
        "[C%04d] %s: stake=$%.2f (f*=%.4f) | p=%.2f | b=%.2f | banca=$%.2f | pend=$%.2f | "
        "pnl_sess=$%+.2f | U=$%.2f | linear=%d | sym=%s%s",
        cycle_id,
        mode_tag,
        final_stake,
        f_star,
        p,
        b,
        bankroll,
        loss_to_recover,
        rm.total_session_profit,
        float(getattr(rm, "dlambert_unit", 0.0)),
        linear_losses,
        symbol,
        rec_info,
    )


def stop_win_target_reached(rm: Any, *, apply_stop_win: bool) -> bool:
    """Retorna True quando lucro da sessao atingiu o alvo de stop win diario."""
    if not apply_stop_win:
        return False
    target = resolve_stop_win_target(
        rm.config,
        rm.initial_bankroll,
        persisted_target=persisted_session_target(rm),
    )
    if rm.total_session_profit < target:
        return False
    rm.logger.info(f"STOP WIN: Meta de ${target:.2f} atingida. Encerrando operações do dia.")
    return True
