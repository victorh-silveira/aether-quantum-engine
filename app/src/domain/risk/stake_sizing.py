"""Calculo de stake Kelly e Martingale."""

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
    max_pct = resolve_max_stake_pct(kelly_config, conviction)
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
    stake_min: float,
    *,
    last_loss_stake: float = 0.0,
) -> float:
    """Calcula stake de recuperacao integral da perda pendente mais lucro alvo."""
    seed = float(last_loss_stake) if last_loss_stake > 0.0 else float(kelly_base)
    seed = max(seed, float(stake_min))
    raw = (loss_to_recover + seed * payout) / payout if payout > 0 and loss_to_recover > 0 else seed
    cap_stake = bankroll
    if payout > 0:
        max_payout_stake = 10000.00 / (1.0 + payout)
        cap_stake = min(cap_stake, max_payout_stake)
    floor_stake = max(stake_min, bankroll * float(kelly_config.get("min_stake_pct", 0.0)))
    return max(floor_stake, min(raw, cap_stake))


def round_stake(value: float, *, martingale: bool) -> float:
    """Arredonda stake para cima em martingale e para baixo em Kelly."""
    if martingale:
        return math.ceil(value * 100) / 100
    return math.floor(value * 100) / 100


def conviction_stop_win_weight(conviction: float, kelly_config: dict[str, Any]) -> float:
    """Interpola fracao do alvo stop win conforme conviccao do sinal."""
    min_conv = float(kelly_config.get("stop_win_kelly_min_conviction", 0.50))
    strong = float(kelly_config.get("stop_win_kelly_conviction_strong", 0.82))
    lo_frac = float(kelly_config.get("stop_win_kelly_min_fraction", 0.12))
    hi_frac = float(kelly_config.get("stop_win_kelly_max_fraction", 0.38))
    if conviction + 1e-9 < min_conv:
        return 0.0
    if conviction >= strong:
        return hi_frac
    span = max(strong - min_conv, 1e-9)
    t = (conviction - min_conv) / span
    return lo_frac + t * (hi_frac - lo_frac)


def _resolve_stop_win_max_stake_pct(
    risk_config: dict[str, Any],
    kelly_config: dict[str, Any],
    payout: float,
) -> float:
    """Deriva teto de stake Kelly para uma tacada atingir o stop win percentual."""
    explicit = float(kelly_config.get("stop_win_max_stake_pct", 0.0))
    if explicit > 0.0:
        return explicit
    stop_pct = float((risk_config or {}).get("large_account_stop_win_pct", 15.0)) / 100.0
    if payout > 0.0:
        return stop_pct / payout
    return stop_pct


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
    """Escala stake Kelly para lucro restante do stop win diario."""
    if not kelly_config.get("stop_win_kelly_enabled", True):
        return kelly_base
    target = resolve_stop_win_target(risk_config, initial_bankroll)
    remaining = max(0.0, target - float(total_session_profit))
    if remaining <= 0.0 or has_active_contracts:
        return kelly_base
    weight = conviction_stop_win_weight(conviction, kelly_config)
    if weight <= 0.0:
        return kelly_base
    cycles_target = max(1.0, float(kelly_config.get("stop_win_kelly_cycles_target", 1.0)))
    goal_stake = (remaining / payout) * weight / cycles_target if payout > 0.0 else kelly_base
    stop_cap = _resolve_stop_win_max_stake_pct(risk_config, kelly_config, payout)
    kelly_cap = resolve_max_stake_pct(kelly_config, conviction)
    max_allowed = bankroll * max(stop_cap, kelly_cap)
    stop_win_stake = min(goal_stake, max_allowed)
    if stop_win_stake > kelly_base:
        return stop_win_stake
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
    recovery_stake: float,
    loss_to_recover: float,
    kelly_base: float,
    payout: float,
    *,
    last_loss_stake: float = 0.0,
) -> str:
    """Monta sufixo de log com detalhes da stake de recuperacao Martingale."""
    if mode_tag != "MARTINGALE":
        return ""
    seed = last_loss_stake if last_loss_stake > 0.0 else kelly_base
    return f" | MARTINGALE ${recovery_stake:.2f} (pend=${loss_to_recover:.2f}+alvo=${seed * payout:.2f})"
