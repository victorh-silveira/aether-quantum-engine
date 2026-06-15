"""Calculo de stake em modo martingale."""

from typing import Any

from src.domain.risk.stake_sizing import conviction_stop_win_weight, round_stake
from src.domain.risk.stop_win_target import resolve_stop_win_target


def martingale_stake(
    bankroll: float,
    loss_to_recover: float,
    kelly_base: float,
    payout: float,
    kelly_config: dict[str, Any],
    stake_min: float,
    *,
    last_loss_stake: float = 0.0,
) -> float:
    """Calcula stake de recuperacao integral da perda pendente mais lucro alvo."""
    seed = float(last_loss_stake) if last_loss_stake > 0.0 else float(kelly_base)
    seed = max(seed, float(stake_min))
    target_frac = float(kelly_config.get("martingale_target_fraction", 1.0))
    step_frac = float(kelly_config.get("martingale_recovery_step_fraction", 1.0))
    step_frac = max(0.15, min(1.0, step_frac))
    effective_loss = float(loss_to_recover) * step_frac
    profit_target = seed * payout * target_frac
    raw = (effective_loss + profit_target) / payout if payout > 0 and effective_loss > 0 else seed
    if payout > 0 and effective_loss > 0:
        raw = max(raw, effective_loss / payout)
    max_mult = float(kelly_config.get("martingale_max_stake_multiplier", 0.0))
    if max_mult > 0.0:
        raw = min(raw, seed * max_mult)
    cap_stake = bankroll
    if payout > 0:
        max_payout_stake = 10000.00 / (1.0 + payout)
        cap_stake = min(cap_stake, max_payout_stake)
    floor_stake = max(stake_min, bankroll * float(kelly_config.get("min_stake_pct", 0.0)))
    return max(floor_stake, min(raw, cap_stake))


def martingale_stop_win_floor(
    bankroll: float,
    payout: float,
    conviction: float,
    risk_config: dict[str, Any],
    kelly_config: dict[str, Any],
    initial_bankroll: float,
    total_session_profit: float,
) -> float:
    """Piso de martingale alinhado ao progresso restante do stop win diario."""
    if not kelly_config.get("stop_win_kelly_enabled", True):
        return 0.0
    progress_frac = float(kelly_config.get("stop_win_martingale_progress_fraction", 0.0))
    if progress_frac <= 0.0 or payout <= 0.0:
        return 0.0
    weight = conviction_stop_win_weight(conviction, kelly_config)
    if weight <= 0.0:
        return 0.0
    target = resolve_stop_win_target(risk_config, initial_bankroll)
    remaining = max(0.0, target - float(total_session_profit))
    if remaining <= 0.0:
        return 0.0
    floor_stake = (remaining / payout) * progress_frac * weight
    cap_stake = bankroll
    max_payout_stake = 10000.00 / (1.0 + payout)
    return min(floor_stake, cap_stake, max_payout_stake)


def resolve_mode_stake(
    *,
    martingale_active: bool,
    bankroll: float,
    loss_to_recover: float,
    kelly_base: float,
    payout: float,
    kelly_config: dict[str, Any],
    stake_min: float,
    last_loss_stake: float = 0.0,
    conviction: float = 0.0,
    risk_config: dict[str, Any] | None = None,
    initial_bankroll: float = 0.0,
    total_session_profit: float = 0.0,
) -> tuple[float, float, str]:
    """Resolve stake final, valor bruto de recuperacao e modo Kelly ou Martingale."""
    if martingale_active:
        recovery = martingale_stake(
            bankroll,
            loss_to_recover,
            kelly_base,
            payout,
            kelly_config,
            stake_min,
            last_loss_stake=last_loss_stake,
        )
        floor_stake = martingale_stop_win_floor(
            bankroll,
            payout,
            conviction,
            risk_config or {},
            kelly_config,
            initial_bankroll,
            total_session_profit,
        )
        recovery = max(recovery, floor_stake)
        return round_stake(recovery, martingale=True), recovery, "MARTINGALE"
    return round_stake(kelly_base, martingale=False), 0.0, "KELLY"


def martingale_log_suffix(
    mode_tag: str,
    final_stake: float,
    loss_to_recover: float,
    kelly_base: float,
    payout: float,
    *,
    last_loss_stake: float = 0.0,
    target_fraction: float = 1.0,
) -> str:
    """Monta sufixo de log com detalhes da stake de recuperacao Martingale."""
    if mode_tag != "MARTINGALE":
        return ""
    seed = last_loss_stake if last_loss_stake > 0.0 else kelly_base
    alvo = seed * payout * float(target_fraction)
    return f" | MARTINGALE ${final_stake:.2f} (pend=${loss_to_recover:.2f}+alvo=${alvo:.2f})"
