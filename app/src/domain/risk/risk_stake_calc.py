"""Calculo de stake Kelly e D'Alembert para RiskManager."""

from typing import Any

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
from src.domain.risk.risk_recovery_state import clear_dust_pending_loss
from src.domain.risk.risk_stake_calc_helpers import cap_final_stake, resolve_f_star_and_kelly_base
from src.domain.risk.risk_stake_flow import (
    apply_stop_win_kelly_boost as _apply_stop_win_kelly_boost,
    apply_target_proximity_to_kelly as _apply_target_proximity_to_kelly,
    emit_cycle_stake_log as _emit_cycle_stake_log,
    stop_win_target_reached as _stop_win_target_reached,
)
from src.domain.risk.stake_sizing import (
    finalize_stake_with_min,
    resolve_stake_conviction,
    resolve_stake_regime,
)


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
    """Retorna True apenas no veto de stop win; Drift Bias Lock desativado."""
    _ = (symbol, kwargs)
    return bool(_stop_win_target_reached(rm, apply_stop_win=apply_stop_win))


def _calculate_kelly_fraction(
    rm: Any,
    symbol: str,
    conviction: float,
    dl_metrics: dict | None,
) -> tuple[float, float, float]:
    """Calcula fracao Kelly bruta e retorna (f, payout b, probabilidade p)."""
    b = float(rm.risk_params.get("payout_estimate", 0.95))
    metrics = dl_metrics if isinstance(dl_metrics, dict) else None
    p = rm.effective_win_rate(symbol, conviction, metrics=metrics)
    kelly_f = (b * p - (1.0 - p)) / b if b > 0 else 0.0
    return kelly_f, b, p


def _resolve_recovery_flags(
    rm: Any,
    symbol: str,
    conviction: float,
    loss_to_recover: float,
    *,
    squeeze_sovereignty: bool,
    kwargs: dict,
) -> tuple[bool, bool, bool, int]:
    """Resolve flags de recovery e perdas lineares para sizing."""
    recovery_active = rm._recovery_allowed(symbol, conviction, **kwargs)
    recovery_financial = bool(loss_to_recover)
    linear_losses = int(getattr(rm, "consecutive_losses_linear", 0))
    recovery_stress = (recovery_financial or linear_losses > 0) and not squeeze_sovereignty
    recovery_bypass_consensus = float(loss_to_recover) > 0.0 and not squeeze_sovereignty
    return recovery_active, recovery_stress, recovery_bypass_consensus, linear_losses


def _apply_mandatory_weak_explore_cap(
    final_stake: float,
    bankroll: float,
    *,
    stake_regime: str,
    mandatory_flag: bool,
    dl_execute: bool,
    kelly_config: dict[str, Any],
) -> float:
    """Aplica teto exploratorio de mandatory fraco apenas em EXPLORE."""
    if stake_regime != "EXPLORE" or not mandatory_flag or dl_execute:
        return final_stake
    weak_pct = float(kelly_config.get("mandatory_weak_max_stake_pct", 0.0) or 0.0)
    if weak_pct <= 0.0:
        return final_stake
    return min(final_stake, max(0.0, float(bankroll) * weak_pct))


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
    """Calcula stake: Kelly em EXPLORE; Soft Recovery amortizado em RECOVER."""
    if check_stake_preconditions_veto(symbol, apply_stop_win=apply_stop_win, rm=rm, kwargs=kwargs):
        return 0.0
    clear_dust_pending_loss(rm)
    loss_to_recover = sum(rm.pending_loss.values())
    linear_preview = int(getattr(rm, "consecutive_losses_linear", 0))
    stake_regime = resolve_stake_regime(pending_loss=loss_to_recover, consecutive_losses_linear=linear_preview)
    if bankroll <= 100.0 and getattr(rm, "dlambert_unit", 0.0) <= 0.0:
        rm.dlambert_unit = 1.00

    dl_metrics = kwargs.get("dl_metrics")
    conviction = resolve_stake_conviction(_metrics_for_conviction(dl_metrics, conviction), rm.kelly_config)
    if isinstance(dl_metrics, dict):
        dl_metrics["stake_regime"] = stake_regime
        if (
            dl_metrics.get("side_eq_blocked")
            or str(dl_metrics.get("side_eq_action") or "") == "hard_skip"
            or dl_metrics.get("signal_status") == "SKIP"
        ):
            return 0.0
        if (
            dl_metrics.get("meta_veto_mode") == "soft" or dl_metrics.get("signal_status") == "SOFT_VETO"
        ) and not _mandatory_trade_flag(kwargs, rm):
            return 0.0

    weak_cap = float(rm.kelly_config.get("mandatory_weak_conviction_cap", 0.55))
    recovery_min = float(rm.kelly_config.get("recovery_min_conviction", 0.50))
    recover_score_floor = min(weak_cap, recovery_min)
    mandatory_blocked = stake_regime == "RECOVER" and conviction + 1e-9 < recover_score_floor
    if mandatory_blocked and float(loss_to_recover) <= 0.0:
        return 0.0

    kelly_f, b, p = _calculate_kelly_fraction(
        rm, symbol, conviction, dl_metrics if isinstance(dl_metrics, dict) else None
    )
    squeeze_sovereignty = d_squeeze_sovereignty_active(dl_metrics if isinstance(dl_metrics, dict) else None)
    recovery_active, recovery_stress, recovery_bypass_consensus, linear_losses = _resolve_recovery_flags(
        rm, symbol, conviction, loss_to_recover, squeeze_sovereignty=squeeze_sovereignty, kwargs=kwargs
    )
    sizing_conviction = conviction
    if isinstance(dl_metrics, dict) and not dl_metrics.get("execute", True):
        sizing_conviction = min(conviction, weak_cap)
    f_star, kelly_base = resolve_f_star_and_kelly_base(
        rm,
        bankroll=bankroll,
        kelly_f=kelly_f,
        sizing_conviction=sizing_conviction,
        dl_metrics=dl_metrics,
        order_direction=kwargs.get("order_direction"),
        recovery_active=stake_regime == "RECOVER" or bool(loss_to_recover),
        recovery_bypass_consensus=recovery_bypass_consensus,
        silent=silent,
    )
    if isinstance(dl_metrics, dict):
        dl_metrics["f_star"] = f_star
        dl_metrics["kelly_fraction"] = f_star
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
        live_metrics=dl_metrics if isinstance(dl_metrics, dict) else None,
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
        f_star=f_star,
    )
    mandatory_flag = _mandatory_trade_flag(kwargs, rm) and not mandatory_blocked
    final_stake = apply_turbo_edge_stake(final_stake, dl_metrics)
    final_stake, safe_cap = cap_final_stake(
        final_stake,
        bankroll=bankroll,
        conviction=conviction,
        recovery_stress=recovery_stress,
        linear_losses=linear_losses,
        rm=rm,
    )
    final_stake = _apply_mandatory_weak_explore_cap(
        final_stake,
        bankroll,
        stake_regime=stake_regime,
        mandatory_flag=mandatory_flag,
        dl_execute=dl_execute,
        kelly_config=rm.kelly_config,
    )
    stake_min = float(rm.risk_params.get("stake_min", 1.0))
    final_stake = finalize_stake_with_min(
        final_stake,
        stake_min,
        bankroll,
        conviction,
        recovery_linear=recovery_stress and not mandatory_blocked and final_stake > 0.0,
        mandatory=mandatory_flag,
    )
    if final_stake < stake_min and not mandatory_flag:
        return 0.0
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
    final_stake = enforce_d_squeeze_stake_floor(final_stake, stake_min, dl_metrics, pending_total=loss_to_recover)
    recovery_infeasible = bool(isinstance(dl_metrics, dict) and dl_metrics.get("recovery_infeasible"))
    if recovery_infeasible and not silent:
        rm.logger.info(
            "RECOVERY_INFEASIBLE | pending=%.2f | cap=%.2f | linear=%d | mode=%s",
            float(loss_to_recover),
            float(safe_cap),
            int(linear_losses),
            stake_regime,
        )
    live_wr = dl_metrics.get("live_wr") if isinstance(dl_metrics, dict) else None
    live_n = int(dl_metrics.get("live_n", 0) or 0) if isinstance(dl_metrics, dict) else 0
    if not silent:
        rm.logger.info(
            "KELLY | p=%.4f | live_wr=%s | live_n=%d | f*=%.6f | mode=%s",
            float(p),
            f"{float(live_wr):.4f}" if live_wr is not None else "n/a",
            int(live_n),
            float(f_star),
            stake_regime.lower(),
        )
    _emit_cycle_stake_log(
        rm,
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
        recovery_infeasible=recovery_infeasible,
    )
    return final_stake
