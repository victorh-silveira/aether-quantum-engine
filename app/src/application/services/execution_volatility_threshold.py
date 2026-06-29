"""Thresholds dinamicos de confianca por regime de volatilidade."""

from dataclasses import dataclass

from src.application.services.execution_volatility_bb import (
    squeeze_dynamic_min_edge,
    squeeze_extreme_regime,
)


@dataclass(frozen=True)
class DynamicThresholds:
    """Limiares CALL/PUT e edge minimo ajustados por regime."""

    call_threshold: float
    put_threshold: float
    min_edge: float
    regime_score: float


def _clamp(value: float, lo: float, hi: float) -> float:
    """Limita valor ao intervalo fechado [lo, hi]."""
    return max(lo, min(hi, float(value)))


def _median_tail(values: list[float], lookback: int) -> float:
    """Mediana das ultimas barras disponiveis."""
    if not values:
        return 0.0
    span = values[-lookback:] if lookback > 0 else values
    ordered = sorted(float(v) for v in span)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) * 0.5


def _normalized_vol_signal(current: float, baseline: float) -> float:
    """Normaliza volatilidade atual versus baseline em torno de 0.5."""
    base = max(float(baseline), 1e-8)
    ratio = float(current) / base
    return _clamp(0.5 + (ratio - 1.0) * 0.35, 0.0, 1.0)


def _vol_component(
    *,
    vol_source: str,
    bb_width: float,
    atr_norm: float,
    bb_baseline: float,
    atr_baseline: float,
) -> float:
    """Combina sinais de BB width e ATR conforme vol_source."""
    bb_signal = _normalized_vol_signal(bb_width, bb_baseline)
    atr_signal = _normalized_vol_signal(atr_norm, atr_baseline)
    source = str(vol_source or "blend").strip().lower()
    if source == "bb_width":
        return bb_signal
    if source == "atr_norm":
        return atr_signal
    return (bb_signal + atr_signal) * 0.5


def volatility_regime_score(
    *,
    bb_width: float,
    atr_norm: float,
    adx: float,
    vol_ratio: float,
    bb_width_history: list[float] | None = None,
    atr_norm_history: list[float] | None = None,
    cfg: dict | None = None,
) -> float:
    """Retorna score [0, 1]: alto = compressao/estouro; baixo = direcional limpo."""
    chunk = cfg if isinstance(cfg, dict) else {}
    lookback = max(8, int(chunk.get("baseline_lookback", 48)))
    compressive_pct = float(chunk.get("compressive_bb_percentile", 0.25))
    directional_adx = float(chunk.get("directional_adx_min", 0.22))
    bb_hist = bb_width_history or [float(bb_width)]
    atr_hist = atr_norm_history or [float(atr_norm)]
    bb_base = _median_tail(bb_hist, lookback)
    atr_base = _median_tail(atr_hist, lookback)
    vol_signal = _vol_component(
        vol_source=str(chunk.get("vol_source", "blend")),
        bb_width=float(bb_width),
        atr_norm=float(atr_norm),
        bb_baseline=bb_base,
        atr_baseline=atr_base,
    )
    bb_low = float(bb_width) <= bb_base * max(0.05, compressive_pct * 2.0)
    atr_rising = float(vol_ratio) > 1.0
    directional_clean = float(adx) >= directional_adx and 0.40 <= vol_signal <= 0.60 and not bb_low
    if directional_clean:
        return _clamp(0.20, 0.0, 1.0)
    if bb_low and atr_rising:
        return _clamp(0.78 + vol_signal * 0.12, 0.0, 1.0)
    return _clamp(vol_signal, 0.0, 1.0)


def resolve_dynamic_thresholds(
    *,
    base_call: float,
    base_put: float,
    base_edge: float,
    regime_score: float,
    cfg: dict | None = None,
) -> DynamicThresholds:
    """Interpola deltas de threshold conforme regime de volatilidade."""
    chunk = cfg if isinstance(cfg, dict) else {}
    score = _clamp(float(regime_score), 0.0, 1.0)
    if score >= 0.5:
        scale = (score - 0.5) * 2.0
        call_delta = float(chunk.get("high_regime_call_delta", 0.03)) * scale
        put_delta = -float(chunk.get("high_regime_put_delta", 0.03)) * scale
        edge_delta = float(chunk.get("high_regime_edge_delta", 0.015)) * scale
    else:
        scale = (0.5 - score) * 2.0
        call_delta = float(chunk.get("low_regime_call_delta", -0.02)) * scale
        put_delta = -float(chunk.get("low_regime_put_delta", -0.02)) * scale
        edge_delta = float(chunk.get("low_regime_edge_delta", -0.01)) * scale
    call_threshold = _clamp(float(base_call) + call_delta, 0.51, 0.62)
    put_threshold = _clamp(float(base_put) + put_delta, 0.38, 0.49)
    min_edge = _clamp(float(base_edge) + edge_delta, 0.02, 0.08)
    return DynamicThresholds(
        call_threshold=call_threshold,
        put_threshold=put_threshold,
        min_edge=min_edge,
        regime_score=score,
    )


def resolve_dynamic_threshold_bundle(
    *,
    base_call: float,
    base_put: float,
    base_edge: float,
    bb_width: float,
    atr_norm: float,
    adx: float,
    vol_ratio: float,
    bb_width_history: list[float] | None = None,
    atr_norm_history: list[float] | None = None,
    cfg: dict | None = None,
    symbol: str = "",
    implied_vol_ratio: float = 1.0,
) -> DynamicThresholds | None:
    """Calcula thresholds dinamicos quando habilitado na configuracao."""
    chunk = cfg if isinstance(cfg, dict) else {}
    if not bool(chunk.get("enabled", False)):
        return None
    regime = volatility_regime_score(
        bb_width=bb_width,
        atr_norm=atr_norm,
        adx=adx,
        vol_ratio=vol_ratio,
        bb_width_history=bb_width_history,
        atr_norm_history=atr_norm_history,
        cfg=chunk,
    )
    scale_bb = bool(chunk.get("implied_vol_bb_scale", True))
    squeeze, bb_norm = squeeze_extreme_regime(
        bb_effective=bb_width,
        bb_width_history=bb_width_history,
        vol_ratio=vol_ratio,
        implied_vol_ratio=implied_vol_ratio,
        symbol=symbol,
        scale_enabled=scale_bb,
    )
    thresholds = resolve_dynamic_thresholds(
        base_call=base_call,
        base_put=base_put,
        base_edge=base_edge,
        regime_score=regime,
        cfg=chunk,
    )
    if squeeze:
        edge = squeeze_dynamic_min_edge(
            base_edge=thresholds.min_edge,
            bb_norm=bb_norm,
            squeeze_slope=float(chunk.get("squeeze_edge_slope", 0.025)),
        )
        return DynamicThresholds(
            call_threshold=thresholds.call_threshold,
            put_threshold=thresholds.put_threshold,
            min_edge=edge,
            regime_score=thresholds.regime_score,
        )
    return thresholds
