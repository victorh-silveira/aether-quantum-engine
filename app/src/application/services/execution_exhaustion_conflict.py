"""Deteccao de conflito entre direcao DL e exaustao CMO/RSI."""

from __future__ import annotations

from src.domain.models.trade import TradeDirection


def _gate_cfg(cfg: dict | None) -> dict:
    """Extrai bloco exhaustion_gate da configuracao de execucao."""
    chunk = cfg if isinstance(cfg, dict) else {}
    gate = chunk.get("exhaustion_gate")
    return gate if isinstance(gate, dict) else {}


def exhaustion_conflict_side(metrics: dict, *, cfg: dict | None = None) -> str | None:
    """Retorna lado de exaustao (call/put) quando RSI e CMO estao em extremo."""
    gate = _gate_cfg(cfg)
    if not bool(gate.get("enabled", True)):
        return None
    indicators = metrics.get("indicators") or {}
    rsi = float(indicators.get("rsi", 0.5))
    cmo = float(indicators.get("cmo", 0.0))
    rsi_ob = float(gate.get("rsi_overbought", 0.72))
    rsi_os = float(gate.get("rsi_oversold", 0.28))
    cmo_bull = float(gate.get("cmo_bull", 0.55))
    cmo_bear = float(gate.get("cmo_bear", -0.55))
    if rsi > rsi_ob and cmo > cmo_bull:
        return "put"
    if rsi < rsi_os and cmo < cmo_bear:
        return "call"
    return None


def _dl_side_name(direction: TradeDirection) -> str:
    """Converte TradeDirection para nome lateral call/put."""
    return "call" if direction == TradeDirection.CALL else "put"


def exhaustion_conflict_penalty(
    metrics: dict,
    dl_direction: TradeDirection,
    *,
    cfg: dict | None = None,
) -> tuple[bool, float]:
    """Penalidade quando DL discorda da exaustao de mercado."""
    gate = _gate_cfg(cfg)
    if not bool(gate.get("enabled", True)):
        return False, 0.0
    exhaust_side = exhaustion_conflict_side(metrics, cfg=cfg)
    if exhaust_side is None:
        return False, 0.0
    dl_side = _dl_side_name(dl_direction)
    if dl_side == exhaust_side:
        return False, 0.0
    margin = float(metrics.get("direction_margin", 0.0))
    base = float(gate.get("min_penalty_skip", 0.12))
    penalty = min(0.35, base + margin * 0.5)
    return True, penalty
