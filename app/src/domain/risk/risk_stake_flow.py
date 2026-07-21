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
    live_metrics: dict[str, Any] | None = None,
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
        live_metrics=live_metrics,
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
    stake_regime: str = "EXPLORE",
    safe_cap: float = 0.0,
    recovery_infeasible: bool = False,
) -> None:
    """Persiste contexto de stake do ciclo para a linha EXEC unica apos boleta."""
    _ = (f_star, p, b, rec_info)
    if cycle_id <= 0 or silent:
        return
    tag = str(mode_tag or "KELLY").upper()
    linear_n = max(0, int(linear_losses))
    compact = f"DAL_L{linear_n}" if ("ALEMBERT" in tag or tag.startswith("DAL")) else "KELLY"
    regime = str(stake_regime or "EXPLORE").upper()
    if regime not in ("EXPLORE", "RECOVER"):
        regime = "EXPLORE"
    audit_tag = f"{regime}_{compact}"
    rm._last_stake_audit = {
        "cycle_id": int(cycle_id),
        "mode_tag": audit_tag,
        "stake": float(final_stake),
        "pending": float(loss_to_recover),
        "bankroll": float(bankroll),
        "linear_losses": int(linear_losses),
        "symbol": str(symbol),
        "stake_regime": regime,
        "cap": float(safe_cap),
        "recovery_infeasible": bool(recovery_infeasible),
    }


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
