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
    martingale_active = rm._martingale_allowed(symbol, conviction, **kwargs)
    _log_martingale_block_if_needed(
        rm,
        loss_to_recover,
        martingale_active=martingale_active,
        symbol=symbol,
        conviction=conviction,
        kwargs=kwargs,
    )

    f_star = max(0.0, kelly_f * float(rm.kelly_config.get("fraction", 0.03)))
    stake_min = float(rm.risk_params.get("stake_min", 1.0))
    kelly_raw = bankroll * f_star
    kelly_raw = rm._apply_stop_win_aggressive_stake(bankroll, kelly_raw, apply_stop_win=apply_stop_win)
    kelly_base = clamp_kelly_stake(bankroll, kelly_raw, rm.kelly_config, conviction)

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

    final_stake, recovery_stake, mode_tag = resolve_mode_stake(
        martingale_active=martingale_active,
        bankroll=bankroll,
        loss_to_recover=loss_to_recover,
        kelly_base=kelly_base,
        payout=b,
        kelly_config=rm.kelly_config,
        conviction=conviction,
        stake_min=stake_min,
        stake_max=rm.stake_max,
    )
    final_stake = finalize_stake_with_min(
        final_stake, stake_min, bankroll, conviction, martingale_active=martingale_active
    )
    cycle_id = int(kwargs.get("cycle_id") or 0)
    rec_info = martingale_log_suffix(mode_tag, recovery_stake, loss_to_recover, kelly_base, b)
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
    conviction: float,
    kwargs: dict,
) -> None:
    """Registra motivo quando ha perda pendente mas martingale nao foi autorizado."""
    if loss_to_recover <= 0.0 or martingale_active:
        return
    block = martingale_block_reason(
        pending_loss=rm.pending_loss,
        recovery_threshold=rm.recovery_threshold,
        conviction=conviction,
        symbol=symbol,
        dl_metrics=kwargs.get("dl_metrics"),
        max_val_brier=float(kwargs.get("max_val_brier", 0.28)),
        order_direction=kwargs.get("order_direction"),
        last_loss_symbol=rm.last_loss_symbol,
        last_loss_direction=rm.last_loss_direction,
        recovery_martingale_min_conviction=rm.recovery_martingale_min_conviction,
        recovery_martingale_min_raw=rm.recovery_martingale_min_raw,
        force_on_pending_loss=rm.martingale_force_on_pending_loss,
    )
    cycle_id = int(kwargs.get("cycle_id", 0))
    rm.logger.info(
        "[C%04d] RISK: Martingale bloqueado (%s) | pend=$%.2f | sym=%s",
        cycle_id,
        block or "?",
        loss_to_recover,
        symbol,
    )
