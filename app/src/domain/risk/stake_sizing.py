"""Calculo de stake Kelly e D'Alembert."""

import math
from typing import Any

from src.domain.risk.stop_win_target import resolve_max_stake_pct, resolve_stop_win_target


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
    """Preenche trade_score a partir de raw_prob quando o score calibrado veio zerado."""
    raw_side = raw_side_from_metrics(metrics)
    score = metric_float(metrics, "trade_score", "conviction", default=0.0)
    if score + 1e-9 < min_raw and raw_side + 1e-9 >= min_raw:
        resolved = max(score, raw_side)
        metrics["trade_score"] = resolved
        metrics["conviction"] = resolved


def resolve_stake_conviction(metrics: dict, kelly_config: dict[str, Any] | None = None) -> float:
    """Deriva conviccao de sizing quando o DL bloqueou mas raw_prob ainda tem lado."""
    cfg = kelly_config or {}
    min_raw = float(cfg.get("stake_conviction_min_raw", 0.51))
    min_stop = float(cfg.get("stop_win_kelly_min_conviction", 0.45))
    score = metric_float(metrics, "trade_score", "conviction", default=0.0)
    raw_side = raw_side_from_metrics(metrics)

    resolved = max(score, raw_side) if raw_side >= min_raw else score

    if resolved + 1e-9 >= min_stop:
        return resolved
    if raw_side + 1e-9 >= min_raw:
        return max(resolved, raw_side)
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
    """Escala stake pelo tempo de rodada (contrato M15 ou ciclo do orquestrador)."""
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


def _resolve_stop_win_max_stake_pct(
    risk_config: dict[str, Any],
    kelly_config: dict[str, Any],
    payout: float,
) -> float:
    """Deriva teto de stake Kelly para uma tacada atingir o stop win percentual."""
    if "stop_win_max_stake_pct" in kelly_config:
        return max(0.0, float(kelly_config["stop_win_max_stake_pct"]))
    stop_pct = float((risk_config or {}).get("large_account_stop_win_pct", 1.0)) / 100.0
    if payout > 0.0:
        return stop_pct / payout
    return stop_pct


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
    if goal_stake > kelly_base:
        return goal_stake
    return kelly_base


def apply_symbol_stake_cap(
    final_stake: float,
    bankroll: float,
    symbol: str,
    kelly_config: dict[str, Any],
) -> float:
    """Retorna stake sem teto por simbolo."""
    _ = (bankroll, symbol, kelly_config)
    return final_stake


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
    if conviction >= 0.50 or recovery_linear or mandatory:
        if final_stake < stake_min and bankroll >= stake_min:
            return stake_min
        if final_stake < stake_min:
            return 0.0
    return final_stake


def _consensus_majority_direction(call_votes: int, put_votes: int) -> str | None:
    """Retorna CALL, PUT ou None em empate de votos tecnicos."""
    if call_votes > put_votes:
        return "CALL"
    if put_votes > call_votes:
        return "PUT"
    return None


def _consensus_opposing_magnitude(value: float, *, order_is_call: bool) -> float:
    """Magnitude normalizada quando o indicador aponta contra a ordem."""
    if order_is_call and value < 0.0:
        return min(1.0, abs(float(value)))
    if not order_is_call and value > 0.0:
        return min(1.0, abs(float(value)))
    return 0.0


def _consensus_rsi_opposing_magnitude(rsi: float, *, order_is_call: bool) -> float:
    """Distancia do RSI em relacao ao neutro quando contra a ordem."""
    neutral = 0.5
    rsi_f = float(rsi)
    if order_is_call and rsi_f < neutral:
        return min(1.0, (neutral - rsi_f) * 2.0)
    if not order_is_call and rsi_f > neutral:
        return min(1.0, (rsi_f - neutral) * 2.0)
    return 0.0


def consensus_vote_agreement(call_votes: int, put_votes: int, order_direction: str) -> float:
    """Taxa de concordancia [0, 1] entre ordem e votos microestruturais."""
    total = int(call_votes) + int(put_votes)
    if total <= 0:
        return 1.0
    ord_call = str(order_direction).strip().upper() == "CALL"
    aligned = int(call_votes) if ord_call else int(put_votes)
    return max(0.0, min(1.0, aligned / total))


def consensus_entropy_kelly_retention(
    metrics: dict,
    order_direction: str | None,
    *,
    kelly_config: dict[str, Any] | None = None,
) -> float:
    """Penalidade convexa de consenso: atenua f* quando ord diverge da maioria dos votos."""
    cfg = kelly_config if isinstance(kelly_config, dict) else {}
    if not bool(cfg.get("consensus_penalty_enabled", True)):
        return 1.0
    if not order_direction or not isinstance(metrics, dict):
        return 1.0
    ord_side = str(order_direction).strip().upper()
    if ord_side not in ("CALL", "PUT"):
        return 1.0
    call_votes = int(metrics.get("call_votes", 0))
    put_votes = int(metrics.get("put_votes", 0))
    majority = _consensus_majority_direction(call_votes, put_votes)
    if majority is None or majority == ord_side:
        return 1.0
    indicators = metrics.get("indicators")
    ind = indicators if isinstance(indicators, dict) else {}
    order_is_call = ord_side == "CALL"
    di_opp = _consensus_opposing_magnitude(float(ind.get("di_diff", 0.0)), order_is_call=order_is_call)
    cmo_opp = _consensus_opposing_magnitude(float(ind.get("cmo", 0.0)), order_is_call=order_is_call)
    rsi_opp = _consensus_rsi_opposing_magnitude(float(ind.get("rsi", 0.5)), order_is_call=order_is_call)
    agreement = consensus_vote_agreement(call_votes, put_votes, ord_side)
    divergence = 1.0 - agreement
    exponent = float(cfg.get("consensus_entropy_exponent", 2.0))
    convex_div = divergence ** max(1.0, exponent)
    di_weight = float(cfg.get("consensus_di_weight", 0.30))
    cmo_weight = float(cfg.get("consensus_cmo_weight", 0.30))
    rsi_weight = float(cfg.get("consensus_rsi_weight", 0.25))
    max_cut = float(cfg.get("consensus_max_cut", 0.50))
    penalty = convex_div * (di_weight * di_opp + cmo_weight * cmo_opp + rsi_weight * rsi_opp)
    retention = 1.0 - min(max_cut, penalty)
    floor = float(cfg.get("consensus_min_retention", 1.0 - max_cut))
    return max(floor, min(1.0, retention))


def consensus_entropy_applies_min_stake(retention: float, kelly_config: dict[str, Any] | None) -> bool:
    """True quando consenso baixo exige stake no piso minimo da API."""
    cfg = kelly_config if isinstance(kelly_config, dict) else {}
    if not bool(cfg.get("consensus_penalty_enabled", True)):
        return False
    floor = float(cfg.get("consensus_min_retention", 1.0 - float(cfg.get("consensus_max_cut", 0.50))))
    return float(retention) <= floor + 1e-9
