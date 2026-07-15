"""Calculo de stake Kelly e D'Alembert para RiskManager."""

from typing import Any

from src.domain.models.trade import TradeDirection
from src.domain.risk.consensus_stake_penalty import (
    apply_neutral_edge_kelly_base,
    apply_turbo_edge_stake,
    d_squeeze_sovereignty_active,
    enforce_d_squeeze_stake_floor,
    max_safe_stake_cap,
)
from src.domain.risk.dlambert_sizing import (
    dlambert_log_suffix,
    effective_martingale_base,
    resolve_dlambert_stake,
)
from src.domain.risk.kelly_f_star_adjustments import (
    apply_consensus_entropy_f_star,
    apply_kelly_fraction_scale,
    kelly_base_with_consensus_floor,
)
from src.domain.risk.risk_stake_flow import (
    apply_stop_win_kelly_boost as _apply_stop_win_kelly_boost,
    apply_target_proximity_to_kelly as _apply_target_proximity_to_kelly,
    emit_cycle_stake_log as _emit_cycle_stake_log,
    stop_win_target_reached as _stop_win_target_reached,
)
from src.domain.risk.stake_sizing import (
    clamp_kelly_stake,
    finalize_stake_with_min,
    resolve_stake_conviction,
)
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


def _mandatory_trade_flag(kwargs: dict, rm: Any) -> bool:
    """Indica se o ciclo exige entrada obrigatoria independente de conviccao."""
    return (
        bool(kwargs.get("mandatory_weak_cap"))
        or bool(kwargs.get("mandatory_trade_each_cycle"))
        or bool(rm.config.get("orchestrator", {}).get("execution", {}).get("mandatory_trade_each_cycle", False))
    )


def check_stake_preconditions_veto(symbol: str, *, apply_stop_win: bool, rm: Any, kwargs: dict) -> bool:
    """Retorna True se houver veto de stop win ou Drift Bias Lock."""
    if _stop_win_target_reached(rm, apply_stop_win=apply_stop_win):
        return True

    dl_metrics = kwargs.get("dl_metrics")
    vol_ratio = 0.0
    bb_width_zscore = 0.0
    if isinstance(dl_metrics, dict):
        vol_ratio = float(dl_metrics.get("vol_ratio", 0.0))
        bb_width_zscore = float(dl_metrics.get("bb_width", dl_metrics.get("bb_width_zscore", 0.0)))

    order_direction = kwargs.get("order_direction")
    dir_name = (
        order_direction.name if isinstance(order_direction, TradeDirection) else str(order_direction or "").upper()
    )

    return ((symbol == "RDBULL" and dir_name == "PUT") or (symbol == "RDBEAR" and dir_name == "CALL")) and (
        vol_ratio >= 2.0 or bb_width_zscore >= 2.0
    )


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
    if check_stake_preconditions_veto(symbol, apply_stop_win=apply_stop_win, rm=rm, kwargs=kwargs):
        return 0.0

    dl_metrics = kwargs.get("dl_metrics")
    conviction = resolve_stake_conviction(_metrics_for_conviction(dl_metrics, conviction), rm.kelly_config)

    mandatory = bool(kwargs.get("mandatory_trade_each_cycle", False))
    is_execute = isinstance(dl_metrics, dict) and bool(dl_metrics.get("execute", True))
    _ = (mandatory, is_execute)

    b = float(rm.risk_params.get("payout_estimate", 0.95))
    p = rm.effective_win_rate(symbol, conviction)
    kelly_f = (b * p - (1.0 - p)) / b if b > 0 else 0.0
    loss_to_recover = sum(rm.pending_loss.values())
    squeeze_sovereignty = d_squeeze_sovereignty_active(dl_metrics if isinstance(dl_metrics, dict) else None)
    recovery_active = rm._recovery_allowed(symbol, conviction, **kwargs)
    recovery_financial = bool(loss_to_recover)
    linear_losses = int(getattr(rm, "consecutive_losses_linear", 0))
    recovery_stress = (recovery_financial or linear_losses > 0) and not squeeze_sovereignty
    recovery_bypass_consensus = float(loss_to_recover) > 0.0 and not squeeze_sovereignty

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
    if recovery_bypass_consensus:
        if isinstance(dl_metrics, dict):
            dl_metrics["consensus_entropy_retention"] = 1.0
    else:
        f_star = apply_consensus_entropy_f_star(
            rm,
            f_star,
            dl_metrics,
            kwargs.get("order_direction"),
            silent=silent,
        )
    f_star = apply_kelly_fraction_scale(f_star, dl_metrics)
    stake_min = float(rm.risk_params.get("stake_min", 1.0))
    if recovery_bypass_consensus:
        kelly_base = clamp_kelly_stake(bankroll, bankroll * f_star, rm.kelly_config, sizing_conviction)
    else:
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
    kelly_base = apply_neutral_edge_kelly_base(kelly_base, bankroll, dl_metrics)

    final_stake, mode_tag = resolve_dlambert_stake(
        recovery_active=recovery_stress,
        bankroll=bankroll,
        kelly_base=kelly_base,
        dlambert_config=rm.dlambert_config,
        rm=rm,
        consecutive_losses_linear=linear_losses,
        pending_total=loss_to_recover,
        payout=b,
        dl_metrics=dl_metrics if isinstance(dl_metrics, dict) else None,
    )
    mandatory_flag = _mandatory_trade_flag(kwargs, rm)
    final_stake = apply_turbo_edge_stake(final_stake, dl_metrics)
    final_stake = min(final_stake, max_safe_stake_cap(bankroll, consecutive_losses_linear=linear_losses))
    final_stake = finalize_stake_with_min(
        final_stake,
        stake_min,
        bankroll,
        conviction,
        recovery_linear=recovery_stress,
        mandatory=mandatory_flag,
    )
    cycle_id = int(kwargs.get("cycle_id") or 0)
    log_kelly_base = kelly_base
    if mode_tag == "D'ALEMBERT":
        log_kelly_base = effective_martingale_base(kelly_base, rm, rm.dlambert_config)
    rec_info = dlambert_log_suffix(
        mode_tag,
        final_stake,
        loss_to_recover,
        log_kelly_base,
        dlambert_unit=float(getattr(rm, "dlambert_unit", 0.0)),
        consecutive_losses_linear=linear_losses,
        dlambert_config=rm.dlambert_config,
        bankroll=bankroll,
        payout=b,
    )
    final_stake = enforce_d_squeeze_stake_floor(
        final_stake,
        stake_min,
        dl_metrics,
        pending_total=loss_to_recover,
    )
    _emit_cycle_stake_log(
        rm,
        cycle_id=cycle_id,
        silent=silent,
        mode_tag=mode_tag,
        final_stake=final_stake,
        f_star=f_star,
        p=p,
        b=b,
        bankroll=bankroll,
        loss_to_recover=loss_to_recover,
        linear_losses=linear_losses,
        symbol=symbol,
        rec_info=rec_info,
    )
    return final_stake
