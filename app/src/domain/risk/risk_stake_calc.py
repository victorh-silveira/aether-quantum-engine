"""Calculo de stake Kelly e Martingale para RiskManager."""

from typing import Any

from src.domain.risk.consensus_stake_penalty import consensus_kelly_retention
from src.domain.risk.martingale_sizing import martingale_log_suffix, resolve_mode_stake
from src.domain.risk.stake_sizing import (
    apply_symbol_stake_cap,
    clamp_kelly_stake,
    compute_single_strike_kelly_base,
    finalize_stake_with_min,
    resolve_stake_conviction,
)
from src.domain.risk.stop_win_target import resolve_stop_win_target


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
    martingale_active: bool,
    apply_stop_win: bool,
    silent: bool,
) -> float:
    """Aplica boost de stake Kelly alinhado ao stop win diario."""
    min_stop_conv = float(rm.kelly_config.get("stop_win_kelly_min_conviction", 0.45))
    stop_win_boost_ok = dl_execute or sizing_conviction + 1e-9 >= min_stop_conv
    if not apply_stop_win or martingale_active or not stop_win_boost_ok:
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


def _apply_mandatory_weak_cap(
    kelly_base: float,
    bankroll: float,
    kelly_config: dict,
    kwargs: dict,
    *,
    martingale_active: bool,
) -> float:
    """Limita stake para entradas fracas de execucao obrigatoria."""
    if not kwargs.get("mandatory_weak_cap") or martingale_active:
        return kelly_base
    weak_pct = float(kelly_config.get("mandatory_weak_max_stake_pct", kelly_config.get("max_stake_pct", 0.004)))
    if weak_pct > 0.0:
        return min(kelly_base, bankroll * weak_pct)
    return kelly_base


def _vol_recovery_context(rm: Any, dl_metrics: dict | None) -> tuple[float, int]:
    """Extrai vol_ratio e perdas consecutivas para sizing martingale adaptativo."""
    indicators = (dl_metrics or {}).get("indicators") or {} if isinstance(dl_metrics, dict) else {}
    return float(indicators.get("vol_ratio", 1.0)), int(getattr(rm, "consecutive_losses", 0))


def _apply_kelly_fraction_scale(f_star: float, dl_metrics: dict | None) -> float:
    """Atenua fracao Kelly quando resolver sinaliza execucao defensiva."""
    if not isinstance(dl_metrics, dict):
        return f_star
    frac_scale = float(dl_metrics.get("kelly_fraction_scale", 1.0))
    if frac_scale < 1.0:
        return f_star * max(0.0, frac_scale)
    return f_star


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

    dl_metrics = kwargs.get("dl_metrics")
    vol_ratio, consecutive_losses = _vol_recovery_context(rm, dl_metrics)
    conviction = resolve_stake_conviction(_metrics_for_conviction(dl_metrics, conviction), rm.kelly_config)

    mandatory = bool(kwargs.get("mandatory_trade_each_cycle", False))
    is_execute = isinstance(dl_metrics, dict) and bool(dl_metrics.get("execute", True))
    if mandatory or is_execute:
        conviction = max(conviction, 0.50)

    b = float(rm.risk_params.get("payout_estimate", 0.95))
    p = rm.effective_win_rate(symbol, conviction)
    kelly_f = (b * p - (1.0 - p)) / b if b > 0 else 0.0
    loss_to_recover = sum(rm.pending_loss.values())
    martingale_active = rm._martingale_allowed(symbol, conviction, **kwargs)

    sizing_conviction = conviction
    if isinstance(dl_metrics, dict) and not dl_metrics.get("execute", True):
        cap_conv = float(rm.kelly_config.get("mandatory_weak_conviction_cap", 0.55))
        sizing_conviction = min(conviction, cap_conv)
    f_star = max(0.0, kelly_f * float(rm.kelly_config.get("fraction", 0.03)))
    if isinstance(dl_metrics, dict):
        retention = consensus_kelly_retention(
            dl_metrics,
            kwargs.get("order_direction"),
            kelly_config=rm.kelly_config,
        )
        if retention < 1.0 and not silent:
            rm.logger.debug(
                "KELLY: consensus retention=%.2f ord=%s votes=%d/%d",
                retention,
                kwargs.get("order_direction"),
                int(dl_metrics.get("call_votes", 0)),
                int(dl_metrics.get("put_votes", 0)),
            )
        f_star *= retention
    f_star = _apply_kelly_fraction_scale(f_star, dl_metrics)
    stake_min = float(rm.risk_params.get("stake_min", 1.0))
    kelly_base = clamp_kelly_stake(bankroll, bankroll * f_star, rm.kelly_config, sizing_conviction)
    dl_execute = not isinstance(dl_metrics, dict) or bool(dl_metrics.get("execute", True))
    kelly_base = _apply_stop_win_kelly_boost(
        rm,
        kelly_base=kelly_base,
        bankroll=bankroll,
        payout=b,
        sizing_conviction=sizing_conviction,
        conviction=conviction,
        dl_execute=dl_execute,
        martingale_active=martingale_active,
        apply_stop_win=apply_stop_win,
        silent=silent,
    )

    kelly_base = _apply_mandatory_weak_cap(
        kelly_base,
        bankroll,
        rm.kelly_config,
        kwargs,
        martingale_active=martingale_active,
    )

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
        vol_ratio=vol_ratio,
        consecutive_losses=consecutive_losses,
    )
    if not martingale_active:
        final_stake = apply_symbol_stake_cap(final_stake, bankroll, symbol, rm.kelly_config)
    stake_max = float(getattr(rm, "stake_max", 0.0))
    if martingale_active and stake_max > 0.0:
        final_stake = min(final_stake, stake_max)
    bankroll_cap = float(rm.kelly_config.get("max_bankroll_stake_fraction", 0.92))
    if bankroll_cap > 0.0:
        final_stake = min(final_stake, bankroll * bankroll_cap)
    mandatory = (
        bool(kwargs.get("mandatory_weak_cap"))
        or bool(kwargs.get("mandatory_trade_each_cycle"))
        or bool(rm.config.get("orchestrator", {}).get("execution", {}).get("mandatory_trade_each_cycle", False))
    )
    final_stake = finalize_stake_with_min(
        final_stake,
        stake_min,
        bankroll,
        conviction,
        martingale_active=martingale_active,
        mandatory=mandatory,
    )
    if martingale_active and final_stake > 0:
        rm.last_martingale_stake = float(final_stake)
    cycle_id = int(kwargs.get("cycle_id") or 0)
    rec_info = martingale_log_suffix(
        mode_tag,
        final_stake,
        loss_to_recover,
        kelly_base,
        b,
        last_loss_stake=float(getattr(rm, "last_loss_stake", 0.0)),
        target_fraction=float(rm.kelly_config.get("martingale_target_fraction", 1.0)),
        vol_ratio=vol_ratio,
        consecutive_losses=consecutive_losses,
        kelly_config=rm.kelly_config,
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
