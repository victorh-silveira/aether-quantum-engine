"""Calculo de stake Kelly e D'Alembert."""

import math
from typing import Any

from src.domain.risk.stake_sizing_consensus import (
    consensus_entropy_applies_min_stake,
    consensus_entropy_kelly_retention,
    consensus_vote_agreement,
)
from src.domain.risk.stop_win_target import resolve_max_stake_pct, resolve_stop_win_target


__all__ = [
    "clamp_kelly_stake",
    "compute_single_strike_kelly_base",
    "consensus_entropy_applies_min_stake",
    "consensus_entropy_kelly_retention",
    "consensus_vote_agreement",
    "conviction_stop_win_weight",
    "enrich_metrics_conviction",
    "finalize_stake_with_min",
    "metric_float",
    "raw_side_from_metrics",
    "resolve_cycle_stake_scale",
    "resolve_stake_conviction",
    "resolve_stake_regime",
    "round_stake",
]


def metric_float(metrics: dict | None, *keys: str, default: float = 0.0) -> float:
    """Le float da primeira chave numerica valida em metrics, ignorando None."""
    if not isinstance(metrics, dict):
        return float(default)
    for key in keys:
        raw = metrics.get(key)
        if raw is None:
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return float(default)


def raw_side_from_metrics(metrics: dict) -> float:
    """Retorna conviccao lateralizada max(p, 1-p) das metricas."""
    raw = metrics.get("raw_prob")
    if raw is None:
        raw = metrics.get("raw_conviction")
    if raw is None:
        return 0.0
    return max(float(raw), 1.0 - float(raw))


def enrich_metrics_conviction(metrics: dict, *, min_raw: float = 0.51) -> None:
    """Preenche trade_score a partir de raw_prob ou senior_trader_conviction quando o score calibrado veio zerado."""
    raw_side = raw_side_from_metrics(metrics)
    senior_conv = metric_float(metrics, "senior_trader_conviction", default=0.0)
    score = metric_float(metrics, "trade_score", "conviction", default=0.0)
    best_conv = max(score, raw_side, senior_conv)
    if best_conv + 1e-9 >= min_raw:
        metrics["trade_score"] = best_conv
        metrics["conviction"] = best_conv


def resolve_stake_regime(*, pending_loss: float, consecutive_losses_linear: int) -> str:
    """Resolve modo EXPLORE ou RECOVER a partir do passivo financeiro."""
    if float(pending_loss) > 0.0 or int(consecutive_losses_linear) >= 1:
        return "RECOVER"
    return "EXPLORE"


def resolve_stake_conviction(metrics: dict, kelly_config: dict[str, Any] | None = None) -> float:
    """Deriva conviccao de sizing considerando DL, raw_prob e senior_trader_conviction."""
    cfg = kelly_config or {}
    min_raw = float(cfg.get("stake_conviction_min_raw", 0.51))
    min_stop = float(cfg.get("stop_win_kelly_min_conviction", 0.45))
    score = metric_float(metrics, "trade_score", "conviction", default=0.0)
    senior_conv = metric_float(metrics, "senior_trader_conviction", default=0.0)
    raw_side = raw_side_from_metrics(metrics)

    best_conv = max(score, raw_side, senior_conv)
    resolved = best_conv if best_conv >= min_raw else score

    if resolved + 1e-9 >= min_stop:
        return resolved
    if best_conv + 1e-9 >= min_raw:
        return max(resolved, best_conv)
    return resolved


def clamp_kelly_stake(
    bankroll: float,
    raw_stake: float,
    kelly_config: dict[str, Any],
    conviction: float,
) -> float:
    """Aplica piso e teto percentual da banca na stake Kelly."""
    min_pct = float(kelly_config.get("min_stake_pct", 0.0))
    floor_stake = bankroll * min_pct if min_pct > 0 else 0.0
    bounded = max(floor_stake, raw_stake)
    max_pct = resolve_max_stake_pct(kelly_config, conviction)
    bankroll_cap = float(kelly_config.get("max_bankroll_stake_fraction", max_pct))
    ceiling_pct = min(float(max_pct), float(bankroll_cap)) if bankroll_cap > 0 else float(max_pct)
    if ceiling_pct > 0.0:
        bounded = min(bounded, float(bankroll) * ceiling_pct)
    return bounded if bounded > 0 else 0.0


def round_stake(value: float, *, recovery_linear: bool) -> float:
    """Arredonda stake para cima em recovery linear e para baixo em Kelly."""
    if recovery_linear:
        return math.ceil(value * 100) / 100
    return math.floor(value * 100) / 100


def resolve_cycle_stake_scale(kelly_config: dict[str, Any], risk_config: dict[str, Any]) -> float:
    """Escala stake pelo tempo de rodada (contrato M2 / ciclo do orquestrador)."""
    if not kelly_config.get("cycle_stake_scale_enabled", True):
        return 1.0
    baseline = float(kelly_config.get("cycle_stake_baseline_seconds", 60))
    if baseline <= 0.0:
        return 1.0
    if kelly_config.get("cycle_stake_use_contract_duration", False):
        params = (risk_config or {}).get("params") or {}
        interval = _contract_duration_seconds(params)
    else:
        orch = (risk_config or {}).get("orchestrator") or {}
        interval = float(orch.get("cycle_interval_seconds", baseline))
    if interval <= baseline:
        return 1.0
    exponent = float(kelly_config.get("cycle_stake_exponent", 0.55))
    return (interval / baseline) ** exponent


def _contract_duration_seconds(params: dict[str, Any]) -> float:
    """Converte duracao do contrato em segundos para escala de stake."""
    dur = max(1, int(params.get("duration", 300)))
    unit = str(params.get("duration_unit", "s")).lower().strip()
    if unit == "m":
        return float(dur * 60)
    if unit == "t":
        return float(dur * 2)
    if unit == "s":
        return float(dur)
    if unit == "d":
        return float(dur * 86400)
    return float(dur * 60)


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


def compute_single_strike_kelly_base(
    kelly_base: float,
    _bankroll: float,
    payout: float,
    conviction: float,
    risk_config: dict[str, Any],
    kelly_config: dict[str, Any],
    initial_bankroll: float,
    total_session_profit: float,
    *,
    has_active_contracts: bool,
    live_metrics: dict[str, Any] | None = None,
) -> float:
    """Escala stake Kelly para lucro restante do stop win diario."""
    if not kelly_config.get("stop_win_kelly_enabled", True):
        return kelly_base
    bag = live_metrics if isinstance(live_metrics, dict) else {}
    live_n = int(bag.get("live_n", 0) or 0)
    live_wr = bag.get("live_wr")
    try:
        wr_ok = live_wr is not None and float(live_wr) >= 0.52
    except (TypeError, ValueError):
        wr_ok = False
    live_n_min = max(0, int(kelly_config.get("stop_win_kelly_live_n_min", 40)))
    if live_n < live_n_min:
        return kelly_base
    if live_n > 0 and not wr_ok:
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
    if goal_stake > kelly_base:
        return goal_stake
    return kelly_base


def finalize_stake_with_min(
    final_stake: float,
    stake_min: float,
    bankroll: float,
    conviction: float,
    *,
    recovery_linear: bool,
    mandatory: bool = False,
) -> float:
    """Garante stake minima ou zero quando conviccao, recovery ou execucao obrigatoria exigem entrada."""
    if float(bankroll) + 1e-12 < float(stake_min):
        return 0.0
    if final_stake <= 0.0 and not mandatory:
        return 0.0
    if conviction >= 0.50 or recovery_linear or mandatory:
        if final_stake < stake_min and bankroll >= stake_min:
            return stake_min
        if final_stake < stake_min:
            return 0.0
    return final_stake
