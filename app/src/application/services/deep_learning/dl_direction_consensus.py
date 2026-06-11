"""Consenso ponderado entre DL, fluxo multi-barra, vela atual e indicadores."""

from src.application.services.deep_learning.dl_candle_flow import (
    binary_direction_vote,
    flow_implied_direction,
    flow_strength,
    sma_extreme_direction,
)
from src.domain.models.trade import TradeDirection


def _dl_direction_from_metrics(metrics: dict) -> TradeDirection | None:
    """Infere CALL/PUT a partir de raw_prob quando disponivel."""
    raw = metrics.get("raw_prob")
    if raw is not None:
        return TradeDirection.CALL if float(raw) > 0.5 else TradeDirection.PUT
    return None


def _raw_side(metrics: dict) -> float:
    """Retorna conviccao lateralizada max(p, 1-p) do modelo."""
    raw = metrics.get("raw_prob")
    if raw is None:
        return 0.0
    return max(float(raw), 1.0 - float(raw))


def _trade_score(metrics: dict) -> float:
    """Extrai score calibrado do candidato."""
    return float(metrics.get("trade_score", metrics.get("conviction", 0.0)))


def _candle_bar_direction(ctx: dict) -> TradeDirection | None:
    """Infere direcao pela vela atual quando corpo e fechamento concordam."""
    if not ctx:
        return None
    body = float(ctx.get("body", 0.0))
    close_loc = float(ctx.get("close_loc", 0.5))
    if body > 0.0 and close_loc + 1e-9 >= 0.48:
        return TradeDirection.CALL
    if body < 0.0 and close_loc <= 0.52 + 1e-9:
        return TradeDirection.PUT
    return None


def _add_vote(weights: tuple[float, float], direction: TradeDirection, weight: float) -> tuple[float, float]:
    """Acumula peso de voto CALL ou PUT."""
    call_w, put_w = weights
    if direction == TradeDirection.CALL:
        return call_w + weight, put_w
    return call_w, put_w + weight


def _dl_vote_weight(score: float, raw_side: float) -> float:
    """Calcula peso do voto DL conforme score e conviccao bruta."""
    base = 0.10 + score * 0.40
    if raw_side + 1e-9 >= 0.62:
        return base + 0.30
    if raw_side + 1e-9 >= 0.55:
        return base + 0.18
    if raw_side + 1e-9 < 0.53:
        return base * 0.35
    return base * 0.55


def resolve_consensus_direction(
    metrics: dict,
    ctx: dict,
    dl_dir: TradeDirection | None = None,
) -> tuple[TradeDirection | None, float]:
    """Resolve CALL/PUT por votos ponderados de DL, fluxo, vela e indicadores."""
    call_w = 0.0
    put_w = 0.0
    score = _trade_score(metrics)
    raw_side = _raw_side(metrics)
    model_dir = _dl_direction_from_metrics(metrics) or dl_dir
    dl_vote_dir = model_dir
    if raw_side + 1e-9 < 0.53 and dl_dir is not None:
        dl_vote_dir = dl_dir
    if dl_vote_dir is not None and (raw_side + 1e-9 >= 0.53 or score + 1e-9 >= 0.48):
        call_w, put_w = _add_vote((call_w, put_w), dl_vote_dir, _dl_vote_weight(score, raw_side))
    flow_dir = flow_implied_direction(ctx)
    flow_s = flow_strength(ctx)
    if flow_dir is not None and flow_s + 1e-9 >= 0.10:
        call_w, put_w = _add_vote((call_w, put_w), flow_dir, flow_s * 0.28)
    candle_dir = _candle_bar_direction(ctx)
    if candle_dir is not None:
        call_w, put_w = _add_vote((call_w, put_w), candle_dir, 0.15)
    indicator_dir = binary_direction_vote(ctx)
    if indicator_dir is not None:
        ind_weight = 0.20 if raw_side + 1e-9 < 0.55 else 0.14
        call_w, put_w = _add_vote((call_w, put_w), indicator_dir, ind_weight)
    sma_z = float(ctx.get("sma_z", 0.0)) if ctx else 0.0
    mr_dir = sma_extreme_direction(sma_z)
    if mr_dir is not None:
        vr = float(ctx.get("variance_ratio", 1.0))
        mr_weight = 0.24 if vr <= 0.85 else 0.14
        if raw_side + 1e-9 < 0.55:
            mr_weight += 0.06
        call_w, put_w = _add_vote((call_w, put_w), mr_dir, mr_weight)
    total = call_w + put_w
    if total < 1e-9:
        return dl_dir or model_dir, 0.0
    strength = abs(call_w - put_w) / total
    if call_w > put_w:
        return TradeDirection.CALL, strength
    if put_w > call_w:
        return TradeDirection.PUT, strength
    return dl_dir or model_dir, 0.0


def dl_direction_from_raw(metrics: dict) -> TradeDirection | None:
    """Expoe direcao pura do modelo a partir de raw_prob."""
    return _dl_direction_from_metrics(metrics)
