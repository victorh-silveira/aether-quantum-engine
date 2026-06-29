"""Hard gate de exaustao RSI+CMO+Keltner com atenuacao severa do peso DL."""

from __future__ import annotations

from src.domain.models.trade import TradeDirection


def _gate_cfg(cfg: dict | None) -> dict:
    """Extrai bloco exhaustion_gate da configuracao de execucao."""
    chunk = cfg if isinstance(cfg, dict) else {}
    gate = chunk.get("exhaustion_gate")
    return gate if isinstance(gate, dict) else {}


def _indicators(metrics: dict) -> dict:
    """Retorna dict de indicadores tecnicos do candidato."""
    raw = metrics.get("indicators")
    return raw if isinstance(raw, dict) else {}


def severe_buy_exhaustion(metrics: dict, *, cfg: dict | None = None) -> bool:
    """True quando RSI, CMO e Keltner indicam exaustao severa de compra."""
    gate = _gate_cfg(cfg)
    if not bool(gate.get("hard_gate_enabled", True)):
        return False
    ind = _indicators(metrics)
    rsi = float(ind.get("rsi", 0.5))
    cmo = float(ind.get("cmo", 0.0))
    keltner = float(ind.get("keltner", 0.5))
    return (
        rsi > float(gate.get("rsi_overbought", 0.73))
        and cmo > float(gate.get("cmo_bull", 0.48))
        and keltner > float(gate.get("keltner_overbought", 1.15))
    )


def severe_sell_exhaustion(metrics: dict, *, cfg: dict | None = None) -> bool:
    """True quando RSI, CMO e Keltner indicam exaustao severa de venda."""
    gate = _gate_cfg(cfg)
    if not bool(gate.get("hard_gate_enabled", True)):
        return False
    ind = _indicators(metrics)
    rsi = float(ind.get("rsi", 0.5))
    cmo = float(ind.get("cmo", 0.0))
    keltner = float(ind.get("keltner", 0.5))
    return (
        rsi < float(gate.get("rsi_oversold", 0.27))
        and cmo < float(gate.get("cmo_bear", -0.48))
        and keltner < float(gate.get("keltner_oversold", -0.15))
    )


def adx_super_trend_exempt(metrics: dict, *, cfg: dict | None = None) -> bool:
    """True quando ADX indica super-tendencia estrutural."""
    gate = _gate_cfg(cfg)
    adx_min = float(gate.get("adx_super_trend_min", 0.40))
    adx = float(_indicators(metrics).get("adx", 0.0))
    return adx > adx_min


def dl_weight_retention(
    metrics: dict,
    dl_direction: TradeDirection,
    *,
    cfg: dict | None = None,
) -> float:
    """Retorna fator de retencao do peso DL (0.20 = atenuacao de 80%)."""
    gate = _gate_cfg(cfg)
    if not bool(gate.get("hard_gate_enabled", True)):
        return 1.0
    if adx_super_trend_exempt(metrics, cfg=cfg):
        return 1.0
    retention = float(gate.get("dl_weight_retention", 0.20))
    if dl_direction == TradeDirection.CALL and severe_buy_exhaustion(metrics, cfg=cfg):
        return retention
    if dl_direction == TradeDirection.PUT and severe_sell_exhaustion(metrics, cfg=cfg):
        return retention
    return 1.0


def hard_gate_score_penalty(*, cfg: dict | None = None) -> float:
    """Penalidade composta para forcar SKIP no quality gate."""
    gate = _gate_cfg(cfg)
    base = float(gate.get("min_penalty_skip", 0.12))
    extra = float(gate.get("hard_gate_score_penalty", 0.25))
    return min(0.50, base + extra)
