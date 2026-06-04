"""Calculo de stake Kelly e Martingale para RiskManager."""

from typing import Any

from src.domain.risk.martingale_gate import martingale_block_reason
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
    rm._prev_martingale_stake = float(rm.last_martingale_stake)
    martingale_active = rm._martingale_allowed(symbol, conviction, **kwargs)
    _log_martingale_block_if_needed(
        rm,
        loss_to_recover,
        martingale_active=martingale_active,
        symbol=symbol,
        kwargs=kwargs,
    )

    dl_metrics = kwargs.get("dl_metrics")
    sizing_conviction = conviction
    if isinstance(dl_metrics, dict) and not dl_metrics.get("execute", True):
        cap_conv = float(rm.kelly_config.get("mandatory_weak_conviction_cap", 0.55))
        sizing_conviction = min(conviction, cap_conv)
    f_star = max(0.0, kelly_f * float(rm.kelly_config.get("fraction", 0.03)))
    stake_min = float(rm.risk_params.get("stake_min", 1.0))
    kelly_raw = bankroll * f_star
    kelly_raw = rm._apply_stop_win_aggressive_stake(bankroll, kelly_raw, apply_stop_win=apply_stop_win)
    kelly_base = clamp_kelly_stake(bankroll, kelly_raw, rm.kelly_config, sizing_conviction)
    if kwargs.get("mandatory_weak_cap"):
        weak_pct = float(
            rm.kelly_config.get("mandatory_weak_max_stake_pct", rm.kelly_config.get("max_stake_pct", 0.004))
        )
        kelly_base = min(kelly_base, bankroll * weak_pct)

    if apply_stop_win and not martingale_active:
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
        if boosted > kelly_base:
            rm.logger.info(
                "RISK: Ativando modo SINGLE STRIKE (Uma Tacada Só)! Sizing boost de $%.2f para $%.2f",
                kelly_base,
                boosted,
            )
        kelly_base = boosted

    stake_base_for_mode = kelly_base
    if martingale_active:
        ref_conv = float(rm.kelly_config.get("martingale_sizing_conviction", 0.60))
        p_mg = rm.effective_win_rate(symbol, ref_conv)
        f_mg = max(0.0, (b * p_mg - (1.0 - p_mg)) / b) if b > 0 else 0.0
        mg_raw = bankroll * f_mg * float(rm.kelly_config.get("fraction", 0.03))
        stake_base_for_mode = clamp_kelly_stake(bankroll, mg_raw, rm.kelly_config, ref_conv)

    final_stake, recovery_stake, mode_tag = resolve_mode_stake(
        martingale_active=martingale_active,
        bankroll=bankroll,
        loss_to_recover=loss_to_recover,
        kelly_base=stake_base_for_mode,
        payout=b,
        kelly_config=rm.kelly_config,
        conviction=conviction,
        stake_min=stake_min,
        stake_max=rm.stake_max,
        consecutive_losses=int(rm.consecutive_losses),
        last_martingale_stake=float(rm.last_martingale_stake),
        last_loss_stake=float(getattr(rm, "last_loss_stake", 0.0)),
    )
    final_stake = finalize_stake_with_min(
        final_stake, stake_min, bankroll, conviction, martingale_active=martingale_active
    )
    if martingale_active and final_stake > 0:
        rm.last_martingale_stake = float(final_stake)
    cycle_id = int(kwargs.get("cycle_id") or 0)
    mult = float(rm.kelly_config.get("martingale_multiplier", 2.0))
    rec_info = martingale_log_suffix(
        mode_tag,
        recovery_stake,
        loss_to_recover,
        kelly_base,
        b,
        consecutive_losses=int(rm.consecutive_losses),
        last_martingale_stake=float(getattr(rm, "_prev_martingale_stake", 0.0)),
        martingale_multiplier=mult,
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


def _log_martingale_block_if_needed(
    rm: Any,
    loss_to_recover: float,
    *,
    martingale_active: bool,
    symbol: str,
    kwargs: dict,
) -> None:
    """Registra motivo quando ha perda pendente mas martingale nao foi autorizado."""
    if loss_to_recover <= 0.0 or martingale_active or kwargs.get("silent"):
        return
    block = martingale_block_reason(
        pending_loss=rm.pending_loss,
        martingale_native=rm.martingale_native,
        block_repeat_loss=rm.martingale_block_repeat_loss,
        symbol=symbol,
        order_direction=kwargs.get("order_direction"),
        last_loss_symbol=rm.last_loss_symbol,
        last_loss_direction=rm.last_loss_direction,
    )
    cycle_id = int(kwargs.get("cycle_id", 0))
    rm.logger.info(
        "[C%04d] RISK: Martingale bloqueado (%s) | pend=$%.2f | sym=%s",
        cycle_id,
        block or "?",
        loss_to_recover,
        symbol,
    )
