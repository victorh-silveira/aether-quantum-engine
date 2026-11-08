"""Penalidade convexa de consenso tecnico sobre retencao Kelly."""

from typing import Any


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
