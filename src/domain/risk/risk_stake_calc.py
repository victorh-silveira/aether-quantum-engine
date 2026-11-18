"""Calculo de stake Kelly e D'Alembert para RiskManager."""

from types import MappingProxyType
from typing import Any

from src.domain.models.signal_evaluation import SignalEvaluation, StakeDecision
from src.domain.risk.consensus_stake_penalty import (
    apply_neutral_edge_kelly_base,
    apply_turbo_edge_stake,
    d_squeeze_sovereignty_active,
    enforce_d_squeeze_stake_floor,
)
from src.domain.risk.dlambert_sizing import (
    dlambert_log_suffix,
    effective_soft_recovery_base,
    resolve_dlambert_stake,
)
from src.domain.risk.kelly_f_star_adjustments import collect_f_star_telemetry
from src.domain.risk.kelly_p_align import calculate_kelly_fraction
from src.domain.risk.risk_recovery_state import clear_dust_pending_loss
from src.domain.risk.risk_stake_calc_helpers import (
    apply_mandatory_weak_explore_cap,
    apply_post_kelly_stake_caps,
    cap_final_stake,
    cap_provisional_stake_with_flag,
    resolve_f_star_and_kelly_base,
)
from src.domain.risk.risk_stake_flow import (
    apply_stop_win_kelly_boost as _apply_stop_win_kelly_boost,
    apply_target_proximity_to_kelly as _apply_target_proximity_to_kelly,
    emit_stake_decision_logs as _emit_stake_decision_logs,
    mandatory_trade_flag as _mandatory_trade_flag,
    metrics_for_conviction as _metrics_for_conviction,
    resolve_recovery_flags as _resolve_recovery_flags,
    stop_win_target_reached as _stop_win_target_reached,
)
from src.domain.risk.soft_recovery_config import pending_waives_scale_explore
from src.domain.risk.stake_sizing import (
    enforce_min_stake_pct,
    finalize_stake_with_min,
    resolve_stake_conviction,
    resolve_stake_regime,
)


_apply_mandatory_weak_explore_cap = apply_mandatory_weak_explore_cap


def check_stake_preconditions_veto(_symbol: str, *, apply_stop_win: bool, rm: Any, **_kwargs) -> bool:
    """Retorna True apenas no veto de stop win; Drift Bias Lock desativado."""
    return bool(_stop_win_target_reached(rm, apply_stop_win=apply_stop_win))


def calculate_stake_decision_for_manager(
    rm: Any,
    bankroll: float,
    symbol: str,
    conviction: float,
    *,
    silent: bool,
    apply_stop_win: bool,
    kwargs: dict,
) -> StakeDecision:
    """Calcula decisao imutavel de stake sem mutar metricas de entrada."""
    if check_stake_preconditions_veto(symbol, apply_stop_win=apply_stop_win, rm=rm, kwargs=kwargs):
        return StakeDecision(stake=0.0)
    clear_dust_pending_loss(rm)
    loss_to_recover = sum(rm.pending_loss.values())
    linear_preview = int(getattr(rm, "consecutive_losses_linear", 0))
    stake_regime = resolve_stake_regime(pending_loss=loss_to_recover, consecutive_losses_linear=linear_preview)
    if bankroll <= 100.0 and getattr(rm, "dlambert_unit", 0.0) <= 0.0:
        rm.dlambert_unit = 1.00

    raw_metrics = kwargs.get("dl_metrics")
    dl_eval = SignalEvaluation.from_metrics(raw_metrics) if raw_metrics is not None else None
    metrics_ro = dl_eval.to_dict() if dl_eval is not None else None
    kelly_fraction_scale = dl_eval.kelly_fraction_scale if dl_eval is not None else 1.0
    conviction = resolve_stake_conviction(_metrics_for_conviction(dl_eval, conviction), rm.kelly_config)
    if dl_eval is not None:
        soft = getattr(rm, "soft_recovery_config", None)
        if dl_eval.scale_force_explore and not pending_waives_scale_explore(
            float(loss_to_recover), soft if isinstance(soft, dict) else None
        ):
            stake_regime = "EXPLORE"
        metrics_ro["stake_regime"] = stake_regime
        status = dl_eval.signal_status.upper()
        if status == "SKIP" or status.startswith("SKIP:") or dl_eval.execution_candidate_ready is False:
            return StakeDecision(stake=0.0, stake_regime=stake_regime, kelly_fraction_scale=kelly_fraction_scale)
        if dl_eval.meta_veto_mode == "soft" or dl_eval.signal_status == "SOFT_VETO":
            if dl_eval.senior_trader_conviction < 0.56 and not _mandatory_trade_flag(kwargs, rm):
                return StakeDecision(stake=0.0, stake_regime=stake_regime, kelly_fraction_scale=kelly_fraction_scale)
            kelly_fraction_scale = kelly_fraction_scale * 0.85
            metrics_ro["kelly_fraction_scale"] = kelly_fraction_scale

    weak_cap = float(rm.kelly_config.get("mandatory_weak_conviction_cap", 0.55))
    recovery_min = float(rm.kelly_config.get("recovery_min_conviction", 0.50))
    recover_score_floor = min(weak_cap, recovery_min)
    recovery_allowed = rm._recovery_allowed(symbol, conviction, **kwargs)
    mandatory_blocked = stake_regime == "RECOVER" and not recovery_allowed and conviction + 1e-9 < recover_score_floor
    if mandatory_blocked and float(loss_to_recover) <= 0.0:
        return StakeDecision(stake=0.0, stake_regime=stake_regime, kelly_fraction_scale=kelly_fraction_scale)

    kelly_f, b, p = calculate_kelly_fraction(
        rm, symbol, conviction, metrics_ro, order_direction=kwargs.get("order_direction")
    )
    squeeze_sovereignty = d_squeeze_sovereignty_active(metrics_ro)
    recovery_active, recovery_stress, recovery_bypass_consensus, linear_losses = _resolve_recovery_flags(
        rm, symbol, conviction, loss_to_recover, squeeze_sovereignty=squeeze_sovereignty, kwargs=kwargs
    )
    sizing_conviction = min(conviction, weak_cap) if (dl_eval is not None and not dl_eval.execute) else conviction
    f_star, kelly_base = resolve_f_star_and_kelly_base(
        rm,
        bankroll=bankroll,
        kelly_f=kelly_f,
        sizing_conviction=sizing_conviction,
        dl_metrics=metrics_ro,
        order_direction=kwargs.get("order_direction"),
        recovery_active=stake_regime == "RECOVER" or bool(loss_to_recover),
        recovery_bypass_consensus=recovery_bypass_consensus,
        silent=silent,
    )
    if metrics_ro is not None:
        metrics_ro["f_star"] = f_star
        metrics_ro["kelly_fraction"] = f_star
    tele = collect_f_star_telemetry(
        rm,
        kelly_f=kelly_f,
        dl_metrics=metrics_ro,
        order_direction=kwargs.get("order_direction"),
        recovery_active=stake_regime == "RECOVER" or bool(loss_to_recover),
        recovery_bypass_consensus=recovery_bypass_consensus,
        silent=silent,
        stake_regime_override=stake_regime,
        kelly_fraction_scale_override=kelly_fraction_scale,
    )
    dl_execute = dl_eval is None or dl_eval.execute
    kelly_base = _apply_stop_win_kelly_boost(
        rm,
        kelly_base=kelly_base,
        bankroll=bankroll,
        payout=b,
        sizing_conviction=sizing_conviction,
        conviction=conviction,
        dl_execute=dl_execute,
        recovery_active=bool(recovery_active) or bool(loss_to_recover) or bool(recovery_stress),
        apply_stop_win=apply_stop_win,
        silent=silent,
        live_metrics=metrics_ro,
    )
    if apply_stop_win:
        kelly_base = _apply_target_proximity_to_kelly(rm, kelly_base, apply_stop_win=True, payout=b)
    kelly_base = apply_neutral_edge_kelly_base(kelly_base, bankroll, metrics_ro)
    final_stake, mode_tag = resolve_dlambert_stake(
        recovery_active=recovery_stress,
        bankroll=bankroll,
        kelly_base=kelly_base,
        dlambert_config=rm.dlambert_config,
        rm=rm,
        consecutive_losses_linear=linear_losses,
        pending_total=loss_to_recover,
        payout=b,
        dl_metrics=metrics_ro,
        f_star=f_star,
    )
    if metrics_ro is not None and bool(metrics_ro.get("recovery_force_explore")):
        stake_regime = "EXPLORE"
        metrics_ro["stake_regime"] = stake_regime
    mandatory_flag = _mandatory_trade_flag(kwargs, rm) and not mandatory_blocked
    final_stake = apply_turbo_edge_stake(final_stake, metrics_ro)
    final_stake, safe_cap = cap_final_stake(
        final_stake,
        bankroll=bankroll,
        conviction=conviction,
        recovery_stress=recovery_stress,
        linear_losses=linear_losses,
        rm=rm,
        metrics=metrics_ro,
    )
    final_stake = apply_mandatory_weak_explore_cap(
        final_stake,
        bankroll,
        stake_regime=stake_regime,
        mandatory_flag=mandatory_flag,
        dl_execute=dl_execute,
        kelly_config=rm.kelly_config,
    )
    stake_min = float(rm.risk_params.get("stake_min", 1.0))
    if mode_tag == "D'ALEMBERT" and final_stake <= stake_min and not mandatory_flag:
        return StakeDecision(
            stake=0.0, f_star=f_star, stake_regime=stake_regime, kelly_fraction_scale=kelly_fraction_scale, **tele
        )
    final_stake = enforce_d_squeeze_stake_floor(final_stake, stake_min, metrics_ro, pending_total=loss_to_recover)
    soft = getattr(rm, "soft_recovery_config", None)
    final_stake = apply_post_kelly_stake_caps(
        final_stake,
        bankroll,
        metrics_ro,
        pending_total=float(loss_to_recover),
        soft_recovery=soft if isinstance(soft, dict) else None,
    )
    final_stake = finalize_stake_with_min(
        final_stake,
        stake_min,
        bankroll,
        conviction,
        recovery_linear=recovery_stress and not mandatory_blocked and final_stake > 0.0,
        mandatory=mandatory_flag,
    )
    final_stake = enforce_min_stake_pct(
        final_stake, bankroll, rm.kelly_config, safe_cap=float(safe_cap), metrics=metrics_ro
    )
    final_stake, provisional_applied = cap_provisional_stake_with_flag(
        final_stake, bankroll, dl_eval, safe_cap=float(safe_cap)
    )
    log_kelly_base = (
        effective_soft_recovery_base(kelly_base, rm, rm.dlambert_config) if mode_tag == "D'ALEMBERT" else kelly_base
    )
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
    _emit_stake_decision_logs(
        rm,
        dl_eval=metrics_ro,
        cycle_id=int(kwargs.get("cycle_id") or 0),
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
        stake_regime=stake_regime,
        safe_cap=safe_cap,
    )
    return StakeDecision(
        stake=final_stake,
        f_star=f_star,
        stake_regime=stake_regime,
        kelly_fraction_scale=kelly_fraction_scale,
        provisional_stake_cap_applied=provisional_applied,
        extra_telemetry=MappingProxyType(metrics_ro or {}),
        **tele,
    )


def calculate_stake_for_manager(
    rm: Any,
    bankroll: float,
    symbol: str,
    conviction: float = 0.5,
    *,
    silent: bool = False,
    apply_stop_win: bool = True,
    kwargs: dict | None = None,
) -> float:
    """Calcula stake numerica delegando a decisao imutavel."""
    return calculate_stake_decision_for_manager(
        rm, bankroll, symbol, conviction, silent=silent, apply_stop_win=apply_stop_win, kwargs=kwargs or {}
    ).stake
