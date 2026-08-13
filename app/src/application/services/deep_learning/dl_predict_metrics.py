"""Anexa metricas dinamicas e de squeeze ao entry DL."""

from src.application.services.deep_learning.dl_congestion import series_last as _series_last
from src.application.services.execution_volatility_bb import squeeze_extreme_regime
from src.application.services.execution_volatility_threshold import DynamicThresholds


def indicators_from_series(series: dict) -> dict[str, float]:
    """Extrai snapshot de indicadores da ultima barra."""
    return {
        "open": _series_last(series, "open"),
        "close": _series_last(series, "close"),
        "high": _series_last(series, "high"),
        "low": _series_last(series, "low"),
        "hurst": _series_last(series, "hurst"),
        "adx": _series_last(series, "adx"),
        "vol_ratio": _series_last(series, "vol_ratio_short_long"),
        "implied_vol_ratio": _series_last(series, "implied_vol_ratio", 1.0),
        "bb_width": _series_last(series, "bb_width"),
        "atr_norm": _series_last(series, "atr_norm"),
        "cmo": _series_last(series, "cmo"),
        "keltner": _series_last(series, "keltner_pct_b"),
        "bb_pct_b": _series_last(series, "bb_pct_b", 0.5),
        "rsi": _series_last(series, "rsi"),
        "macd": _series_last(series, "macd"),
        "macd_sig": _series_last(series, "macd_signal"),
        "di_diff": _series_last(series, "di_diff"),
    }


def attach_dynamic_metrics(
    metrics: dict,
    *,
    dynamic: DynamicThresholds | None,
    bb_width: float,
    vol_ratio: float,
    implied_vol_ratio: float,
    symbol: str,
    bb_history: list[float],
    scale_enabled: bool,
    runtime: dict,
) -> None:
    """Preenche thresholds dinamicos, squeeze e entropia no dict de metricas."""
    if dynamic is not None:
        metrics["dynamic_call_threshold"] = dynamic.call_threshold
        metrics["dynamic_put_threshold"] = dynamic.put_threshold
        metrics["dynamic_min_edge"] = dynamic.min_edge
        metrics["volatility_regime"] = dynamic.regime_score
        squeeze, bb_norm = squeeze_extreme_regime(
            bb_effective=bb_width,
            bb_width_history=bb_history,
            vol_ratio=vol_ratio,
            implied_vol_ratio=implied_vol_ratio,
            symbol=symbol,
            scale_enabled=scale_enabled,
        )
        metrics["squeeze_extreme"] = squeeze
        metrics["bb_norm"] = bb_norm
    runtime_entropy = runtime.get("calibrated_entropy")
    if runtime_entropy is not None:
        metrics["calibrated_entropy"] = float(runtime_entropy)
    if runtime.get("entropy_violation") is not None:
        metrics["entropy_violation"] = bool(runtime.get("entropy_violation"))
