"""Calculo de stake Kelly e Martingale."""

import datetime
import math
from typing import Any

from src.domain.risk.stop_win_target import resolve_max_stake_pct, resolve_stop_win_target


def clamp_kelly_stake(
    bankroll: float,
    raw_stake: float,
    kelly_config: dict[str, Any],
    conviction: float,
) -> float:
    """Limita a stake Kelly entre piso e teto percentuais da banca."""
    max_pct = resolve_max_stake_pct(kelly_config, conviction, is_recovery=False)
    min_pct = float(kelly_config.get("min_stake_pct", 0.0))
    floor_stake = bankroll * min_pct if min_pct > 0 else 0.0
    ceiling = bankroll * max_pct
    bounded = max(floor_stake, min(raw_stake, ceiling))
    return bounded if bounded > 0 else 0.0


def martingale_stake(
    bankroll: float,
    loss_to_recover: float,
    kelly_base: float,
    payout: float,
    kelly_config: dict[str, Any],
    _conviction: float,
    stake_min: float,
    stake_max: float,
    *,
    consecutive_losses: int = 0,
    last_martingale_stake: float = 0.0,
    last_loss_stake: float = 0.0,
) -> float:
    """Calcula stake de recuperacao com progressao nativa (multiplicador sobre ultima entrada)."""
    multiplier = max(1.0, float(kelly_config.get("martingale_multiplier", 2.0)))
    reference = max(float(kelly_base), float(last_loss_stake), float(stake_min))
    profit_target = reference * payout
    cover = (loss_to_recover + profit_target) / payout if payout > 0 else 0.0
    step = max(1, int(consecutive_losses))
    progressive = last_martingale_stake * multiplier if last_martingale_stake > 0.0 else reference * (multiplier**step)
    raw = max(progressive, cover)
    cap_conviction = float(kelly_config.get("martingale_cap_conviction", 0.5))
    max_pct = resolve_max_stake_pct(kelly_config, cap_conviction, is_recovery=True)
    cap_stake = min(bankroll * max_pct, stake_max)
    floor_stake = max(stake_min, bankroll * float(kelly_config.get("min_stake_pct", 0.0)))
    return max(floor_stake, min(raw, cap_stake))


def round_stake(value: float, *, martingale: bool) -> float:
    """Arredonda stake para cima em martingale e para baixo em Kelly."""
    if martingale:
        return math.ceil(value * 100) / 100
    return math.floor(value * 100) / 100


def compute_single_strike_kelly_base(
    kelly_base: float,
    bankroll: float,
    payout: float,
    conviction: float,
    risk_config: dict[str, Any],
    kelly_config: dict[str, Any],
    initial_bankroll: float,
    total_session_profit: float,
    *,
    has_active_contracts: bool,
) -> float:
    """Aplica boost de single strike quando stop win e janela permitem."""
    now_utc = datetime.datetime.now(datetime.UTC)
    in_window = 12 <= now_utc.hour < 17
    target = resolve_stop_win_target(risk_config, initial_bankroll)
    remaining = max(0.0, target - float(total_session_profit))
    if not (in_window and conviction >= 0.75 and remaining > 0 and not has_active_contracts):
        return kelly_base
    goal_stake = remaining / payout
    max_allowed = bankroll * resolve_max_stake_pct(kelly_config, conviction)
    single_strike_stake = min(goal_stake, max_allowed)
    if single_strike_stake > kelly_base:
        return single_strike_stake
    return kelly_base


def finalize_stake_with_min(
    final_stake: float,
    stake_min: float,
    bankroll: float,
    conviction: float,
    *,
    martingale_active: bool,
) -> float:
    """Garante stake minima ou zero quando conviccao ou martingale exigem entrada."""
    if conviction >= 0.50 or martingale_active:
        if final_stake < stake_min and bankroll >= stake_min:
            return stake_min
        if final_stake < stake_min:
            return 0.0
    return final_stake


def resolve_mode_stake(
    *,
    martingale_active: bool,
    bankroll: float,
    loss_to_recover: float,
    kelly_base: float,
    payout: float,
    kelly_config: dict[str, Any],
    conviction: float,
    stake_min: float,
    stake_max: float,
    consecutive_losses: int = 0,
    last_martingale_stake: float = 0.0,
    last_loss_stake: float = 0.0,
) -> tuple[float, float, str]:
    """Resolve stake final, valor bruto de recuperacao e modo Kelly ou Martingale."""
    if martingale_active:
        recovery = martingale_stake(
            bankroll,
            loss_to_recover,
            kelly_base,
            payout,
            kelly_config,
            conviction,
            stake_min,
            stake_max,
            consecutive_losses=consecutive_losses,
            last_martingale_stake=last_martingale_stake,
            last_loss_stake=last_loss_stake,
        )
        return round_stake(recovery, martingale=True), recovery, "MARTINGALE"
    return round_stake(kelly_base, martingale=False), 0.0, "KELLY"


def martingale_log_suffix(
    mode_tag: str,
    recovery_stake: float,
    loss_to_recover: float,
    kelly_base: float,
    payout: float,
    *,
    consecutive_losses: int = 0,
    last_martingale_stake: float = 0.0,
    martingale_multiplier: float = 2.0,
) -> str:
    """Monta sufixo de log com detalhes da stake de recuperacao Martingale."""
    if mode_tag != "MARTINGALE":
        return ""
    step = max(1, int(consecutive_losses))
    prev = f"${last_martingale_stake:.2f}" if last_martingale_stake > 0 else f"base=${kelly_base:.2f}"
    return (
        f" | MARTINGALE x{martingale_multiplier:.2f} passo={step} {prev}"
        f" -> ${recovery_stake:.2f} (pend=${loss_to_recover:.2f}+alvo=${kelly_base * payout:.2f})"
    )
