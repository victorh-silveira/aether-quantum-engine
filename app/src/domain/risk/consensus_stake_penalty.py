"""Modificador de Kelly por divergencia entre ordem e votos tecnicos."""

from __future__ import annotations

from typing import Any


def _majority_direction(call_votes: int, put_votes: int) -> str | None:
    """Retorna CALL, PUT ou None em caso de empate."""
    if call_votes > put_votes:
        return "CALL"
    if put_votes > call_votes:
        return "PUT"
    return None


def _opposing_magnitude(value: float, *, order_is_call: bool) -> float:
    """Magnitude do indicador quando aponta contra a direcao da ordem."""
    if order_is_call and value < 0.0:
        return abs(float(value))
    if not order_is_call and value > 0.0:
        return abs(float(value))
    return 0.0


def consensus_kelly_retention(
    metrics: dict,
    order_direction: str | None,
    *,
    kelly_config: dict[str, Any] | None = None,
) -> float:
    """Retorna fator [0.50, 1.0] para atenuar f* quando ord diverge do consenso tecnico."""
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
    majority = _majority_direction(call_votes, put_votes)
    if majority is None or majority == ord_side:
        return 1.0
    indicators = metrics.get("indicators")
    ind = indicators if isinstance(indicators, dict) else {}
    di_diff = float(ind.get("di_diff", 0.0))
    cmo = float(ind.get("cmo", 0.0))
    order_is_call = ord_side == "CALL"
    di_opp = _opposing_magnitude(di_diff, order_is_call=order_is_call)
    cmo_opp = _opposing_magnitude(cmo, order_is_call=order_is_call)
    di_weight = float(cfg.get("consensus_di_weight", 0.35))
    cmo_weight = float(cfg.get("consensus_cmo_weight", 0.40))
    max_cut = float(cfg.get("consensus_max_cut", 0.50))
    penalty = di_weight * di_opp + cmo_weight * cmo_opp
    retention = 1.0 - min(max_cut, penalty)
    floor = 1.0 - max_cut
    return max(floor, min(1.0, retention))
