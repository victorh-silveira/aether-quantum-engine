"""Anexa metricas dinamicas e de squeeze ao entry DL."""

from src.application.services.execution_volatility_bb import squeeze_extreme_regime
from src.application.services.execution_volatility_threshold import DynamicThresholds


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
