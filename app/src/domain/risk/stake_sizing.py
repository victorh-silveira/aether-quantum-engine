"""Calculo de stake Kelly e Martingale."""

import math
from typing import Any

from src.domain.risk.stop_win_target import resolve_max_stake_pct, resolve_stop_win_target


def raw_side_from_metrics(metrics: dict) -> float:
    """Retorna conviccao lateralizada max(p, 1-p) das metricas."""
    raw = metrics.get("raw_prob")
    if raw is None:
        raw = metrics.get("raw_conviction")
    if raw is None:
        return 0.0
    return max(float(raw), 1.0 - float(raw))


def enrich_metrics_conviction(metrics: dict, *, min_raw: float = 0.51) -> None:
    """Preenche trade_score a partir de raw_prob quando o score calibrado veio zerado."""
    raw_side = raw_side_from_metrics(metrics)
    score = float(metrics.get("trade_score", metrics.get("conviction", 0.0)))
    if score + 1e-9 < min_raw and raw_side + 1e-9 >= min_raw:
        resolved = max(score, raw_side)
        metrics["trade_score"] = resolved
        metrics["conviction"] = resolved


def resolve_stake_conviction(metrics: dict, kelly_config: dict[str, Any] | None = None) -> float:
    """Deriva conviccao de sizing quando o DL bloqueou mas raw_prob ainda tem lado."""
    cfg = kelly_config or {}
    min_raw = float(cfg.get("stake_conviction_min_raw", 0.51))
    min_stop = float(cfg.get("stop_win_kelly_min_conviction", 0.45))
    score = float(metrics.get("trade_score", metrics.get("conviction", 0.0)))
    raw_side = raw_side_from_metrics(metrics)
    if score + 1e-9 >= min_stop:
        return score
    if raw_side + 1e-9 >= min_raw:
        return max(score, raw_side)
    return score


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


def round_stake(value: float, *, martingale: bool) -> float:
    """Arredonda stake para cima em martingale e para baixo em Kelly."""
    if martingale:
        return math.ceil(value * 100) / 100
    return math.floor(value * 100) / 100


def resolve_cycle_stake_scale(kelly_config: dict[str, Any], risk_config: dict[str, Any]) -> float:
    """Escala stake pelo intervalo de ciclo quando timeframe e maior que o baseline."""
    if not kelly_config.get("cycle_stake_scale_enabled", True):
        return 1.0
    baseline = float(kelly_config.get("cycle_stake_baseline_seconds", 60))
    if baseline <= 0.0:
        return 1.0
    orch = (risk_config or {}).get("orchestrator") or {}
    interval = float(orch.get("cycle_interval_seconds", baseline))
    if interval <= baseline:
        return 1.0
    exponent = float(kelly_config.get("cycle_stake_exponent", 0.55))
    return (interval / baseline) ** exponent


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
    stop_pct = float((risk_config or {}).get("large_account_stop_win_pct", 1.0)) / 100.0
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
    cycle_scale = resolve_cycle_stake_scale(kelly_config, risk_config)
    goal_stake = (remaining / payout) * weight / cycles_target * cycle_scale if payout > 0.0 else kelly_base
    stop_cap = _resolve_stop_win_max_stake_pct(risk_config, kelly_config, payout)
    kelly_cap = resolve_max_stake_pct(kelly_config, conviction)
    max_allowed = bankroll * max(stop_cap, kelly_cap)
    stop_win_stake = min(goal_stake, max_allowed)
    if stop_win_stake > kelly_base:
        return stop_win_stake
    return kelly_base


def apply_symbol_stake_cap(
    final_stake: float,
    bankroll: float,
    symbol: str,
    kelly_config: dict[str, Any],
) -> float:
    """Aplica teto percentual da banca por simbolo quando configurado."""
    caps = kelly_config.get("symbol_max_stake_pct")
    if not isinstance(caps, dict) or bankroll <= 0.0:
        return final_stake
    cap_pct = caps.get(str(symbol))
    if cap_pct is None:
        return final_stake
    return min(final_stake, bankroll * float(cap_pct))


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
