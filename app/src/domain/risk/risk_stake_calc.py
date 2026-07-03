"""Calculo de stake Kelly e D'Alembert para RiskManager."""

from typing import Any

from src.domain.risk.dlambert_sizing import (
    dlambert_log_suffix,
    resolve_dlambert_stake,
)
from src.domain.risk.kelly_f_star_adjustments import (
    apply_consensus_entropy_f_star,
    apply_kelly_fraction_scale,
    kelly_base_with_consensus_floor,
)
from src.domain.risk.stake_sizing import (
    compute_single_strike_kelly_base,
    finalize_stake_with_min,
    resolve_stake_conviction,
)
from src.domain.risk.stake_target_proximity import apply_target_proximity_damping
from src.domain.risk.stop_win_target import persisted_session_target, resolve_stop_win_target
from src.domain.risk.super_concordance_kelly import apply_super_concordance_kelly_fraction


def _metrics_for_conviction(dl_metrics: dict | None, conviction: float) -> dict:
    """Monta metricas para resolver conviccao de sizing."""
    if isinstance(dl_metrics, dict):
        merged = dict(dl_metrics)
        if "trade_score" not in merged and "conviction" not in merged:
            merged["trade_score"] = conviction
            merged["conviction"] = conviction
        return merged
    return {"trade_score": conviction, "conviction": conviction}


def _apply_stop_win_kelly_boost(
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


def _apply_target_proximity_to_kelly(rm: Any, kelly_base: float, *, apply_stop_win: bool) -> float:
    """Comprime Kelly bruto pela proximidade da meta de stop win da sessao."""
    if not apply_stop_win:
        return kelly_base
    target = resolve_stop_win_target(
        rm.config,
        rm.initial_bankroll,
        persisted_target=persisted_session_target(rm),
    )
    return apply_target_proximity_damping(kelly_base, target, rm.total_session_profit)


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
    """Calcula stake final com Kelly ou D'Alembert conforme estado do gerenciador."""
    if apply_stop_win:
        target = resolve_stop_win_target(
            rm.config,
            rm.initial_bankroll,
            persisted_target=persisted_session_target(rm),
        )
        if rm.total_session_profit >= target:
            rm.logger.info(f"STOP WIN: Meta de ${target:.2f} atingida. Encerrando operações do dia.")
            return 0.0

    dl_metrics = kwargs.get("dl_metrics")
    conviction = resolve_stake_conviction(_metrics_for_conviction(dl_metrics, conviction), rm.kelly_config)

    mandatory = bool(kwargs.get("mandatory_trade_each_cycle", False))
    is_execute = isinstance(dl_metrics, dict) and bool(dl_metrics.get("execute", True))
    if mandatory or is_execute:
        conviction = max(conviction, 0.50)

    b = float(rm.risk_params.get("payout_estimate", 0.95))
    p = rm.effective_win_rate(symbol, conviction)
    kelly_f = (b * p - (1.0 - p)) / b if b > 0 else 0.0
    loss_to_recover = sum(rm.pending_loss.values())
    recovery_active = rm._recovery_allowed(symbol, conviction, **kwargs)
    recovery_financial = bool(loss_to_recover)

    sizing_conviction = conviction
    if isinstance(dl_metrics, dict) and not dl_metrics.get("execute", True):
        cap_conv = float(rm.kelly_config.get("mandatory_weak_conviction_cap", 0.55))
        sizing_conviction = min(conviction, cap_conv)
    f_star = apply_super_concordance_kelly_fraction(
        kelly_f,
        rm.kelly_config,
        dl_metrics if isinstance(dl_metrics, dict) else None,
        kwargs.get("order_direction"),
        recovery_active=recovery_financial,
    )
    f_star = apply_consensus_entropy_f_star(
        rm,
        f_star,
        dl_metrics,
        kwargs.get("order_direction"),
        silent=silent,
    )
    f_star = apply_kelly_fraction_scale(f_star, dl_metrics)
    stake_min = float(rm.risk_params.get("stake_min", 1.0))
    kelly_base = kelly_base_with_consensus_floor(
        bankroll,
        f_star,
        dl_metrics,
        rm.kelly_config,
        sizing_conviction,
        stake_min,
    )
    dl_execute = not isinstance(dl_metrics, dict) or bool(dl_metrics.get("execute", True))
    kelly_base = _apply_stop_win_kelly_boost(
        rm,
        kelly_base=kelly_base,
        bankroll=bankroll,
        payout=b,
        sizing_conviction=sizing_conviction,
        conviction=conviction,
        dl_execute=dl_execute,
        recovery_active=recovery_active,
        apply_stop_win=apply_stop_win,
        silent=silent,
    )
    if apply_stop_win:
        kelly_base = _apply_target_proximity_to_kelly(rm, kelly_base, apply_stop_win=True)

    linear_losses = int(getattr(rm, "consecutive_losses_linear", 0))
    final_stake, mode_tag = resolve_dlambert_stake(
        recovery_active=recovery_active and recovery_financial,
        bankroll=bankroll,
        kelly_base=kelly_base,
        dlambert_config=rm.dlambert_config,
        rm=rm,
        consecutive_losses_linear=linear_losses,
        pending_total=loss_to_recover,
    )
    mandatory_flag = (
        bool(kwargs.get("mandatory_weak_cap"))
        or bool(kwargs.get("mandatory_trade_each_cycle"))
        or bool(rm.config.get("orchestrator", {}).get("execution", {}).get("mandatory_trade_each_cycle", False))
    )
    final_stake = finalize_stake_with_min(
        final_stake,
        stake_min,
        bankroll,
        conviction,
        recovery_linear=recovery_active,
        mandatory=mandatory_flag,
    )
    cycle_id = int(kwargs.get("cycle_id") or 0)
    rec_info = dlambert_log_suffix(
        mode_tag,
        final_stake,
        loss_to_recover,
        kelly_base,
        dlambert_unit=float(getattr(rm, "dlambert_unit", 0.0)),
        consecutive_losses_linear=linear_losses,
        dlambert_config=rm.dlambert_config,
        bankroll=bankroll,
    )
    if cycle_id > 0 and not silent:
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
    return final_stake
