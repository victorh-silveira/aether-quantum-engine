"""Calculo de stake Kelly e Martingale para RiskManager."""

from typing import Any

from src.domain.risk.stake_sizing import (
    clamp_kelly_stake,
    compute_single_strike_kelly_base,
    finalize_stake_with_min,
    martingale_log_suffix,
    resolve_mode_stake,
)
from src.domain.risk.stop_win_target import resolve_stop_win_target


def calculate_stake_for_manager(
    rm: Any,
    bankroll: float,
    symbol: str,
    conviction: float,
    *,
    silent: bool,
    apply_stop_win: bool,
    kwargs: dict,
) -> float:
    """Calcula stake final com Kelly ou Martingale conforme estado do gerenciador."""
    if apply_stop_win:
        target = resolve_stop_win_target(rm.config, rm.initial_bankroll)
        if rm.total_session_profit >= target:
            rm.logger.info(f"STOP WIN: Meta de ${target:.2f} atingida. Encerrando operações do dia.")
            return 0.0

    b = float(rm.risk_params.get("payout_estimate", 0.95))
    p = rm.effective_win_rate(symbol, conviction)
    kelly_f = (b * p - (1.0 - p)) / b if b > 0 else 0.0
    loss_to_recover = sum(rm.pending_loss.values())
    martingale_active = rm._martingale_allowed(symbol, conviction, **kwargs)

    dl_metrics = kwargs.get("dl_metrics")
    sizing_conviction = conviction
    if isinstance(dl_metrics, dict) and not dl_metrics.get("execute", True):
        cap_conv = float(rm.kelly_config.get("mandatory_weak_conviction_cap", 0.55))
        sizing_conviction = min(conviction, cap_conv)
    f_star = max(0.0, kelly_f * float(rm.kelly_config.get("fraction", 0.03)))
    stake_min = float(rm.risk_params.get("stake_min", 1.0))
    kelly_raw = bankroll * f_star
    kelly_base = clamp_kelly_stake(bankroll, kelly_raw, rm.kelly_config, sizing_conviction)
    stop_win_kelly = bool(rm.kelly_config.get("stop_win_kelly_enabled", True))
    if kwargs.get("mandatory_weak_cap") and not martingale_active and not stop_win_kelly:
        weak_pct = float(
            rm.kelly_config.get("mandatory_weak_max_stake_pct", rm.kelly_config.get("max_stake_pct", 0.004))
        )
        kelly_base = min(kelly_base, bankroll * weak_pct)

    dl_approved = not isinstance(dl_metrics, dict) or bool(dl_metrics.get("execute", True))
    if apply_stop_win and not martingale_active and dl_approved:
        boosted = compute_single_strike_kelly_base(
            kelly_base,
            bankroll,
            b,
            conviction,
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
        kelly_base = boosted

    final_stake, recovery_stake, mode_tag = resolve_mode_stake(
        martingale_active=martingale_active,
        bankroll=bankroll,
        loss_to_recover=loss_to_recover,
        kelly_base=kelly_base,
        payout=b,
        kelly_config=rm.kelly_config,
        stake_min=stake_min,
        last_loss_stake=float(getattr(rm, "last_loss_stake", 0.0)),
        conviction=conviction,
        risk_config=rm.config,
        initial_bankroll=rm.initial_bankroll,
        total_session_profit=rm.total_session_profit,
    )
    if martingale_active and isinstance(dl_metrics, dict):
        trade_score = float(dl_metrics.get("trade_score", conviction))
        consensus = float(dl_metrics.get("consensus_strength", 1.0))
        weak_recovery = trade_score + 1e-9 < 0.55 or consensus + 1e-9 < 0.35
        if weak_recovery:
            final_stake = min(final_stake, bankroll * 0.03)
    final_stake = min(final_stake, bankroll * 0.92)
    final_stake = finalize_stake_with_min(
        final_stake, stake_min, bankroll, conviction, martingale_active=martingale_active
    )
    if martingale_active and final_stake > 0:
        rm.last_martingale_stake = float(final_stake)
    cycle_id = int(kwargs.get("cycle_id") or 0)
    rec_info = martingale_log_suffix(
        mode_tag,
        recovery_stake,
        loss_to_recover,
        kelly_base,
        b,
        last_loss_stake=float(getattr(rm, "last_loss_stake", 0.0)),
    )
    if cycle_id > 0 and not silent:
        rm.logger.info(
            "[C%04d] %s: stake=$%.2f (f*=%.4f) | p=%.2f | b=%.2f | banca=$%.2f | sym=%s%s",
            cycle_id,
            mode_tag,
            final_stake,
            f_star,
            p,
            b,
            bankroll,
            symbol,
            rec_info,
        )
    return final_stake
