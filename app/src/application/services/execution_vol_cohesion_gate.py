"""Coesao de volatilidade macro M15 versus micro M1 para rebaixamento entropico."""

from src.application.services.execution_universal_regime_types import RegimeState


MACRO_VOL_EXPANSION_MIN = 1.15
MICRO_VOL_DIVERGENCE_MAX = 0.50


def macro_vol_ratio(metrics: dict) -> float:
    """Le vol_ratio macro M15 com fallback para indicadores micro."""
    macro = metrics.get("macro_indicators")
    if isinstance(macro, dict) and macro.get("vol_ratio") is not None:
        return float(macro["vol_ratio"])
    indicators = metrics.get("indicators") or {}
    return float(indicators.get("vol_ratio", 1.0))


def micro_vol_ratio(metrics: dict) -> float:
    """Le vol_ratio micro M1 dos indicadores de execucao."""
    indicators = metrics.get("indicators") or {}
    return float(indicators.get("vol_ratio", macro_vol_ratio(metrics)))


def apply_vol_cohesion_entropic_downgrade(metrics: dict) -> bool:
    """Rebaixa para ENTROPIC_NOISE quando M1 comprime sob macro M15 expandido."""
    macro_vol = macro_vol_ratio(metrics)
    micro_vol = micro_vol_ratio(metrics)
    if macro_vol + 1e-9 < MACRO_VOL_EXPANSION_MIN or micro_vol + 1e-9 >= MICRO_VOL_DIVERGENCE_MAX:
        return False
    regime = RegimeState.ENTROPIC_NOISE.value
    metrics["universal_regime"] = regime
    metrics["universal_regime_scenario"] = regime
    metrics["gate_penalty"] = "noise"
    metrics["regime_skip_cycle"] = True
    metrics["vol_cohesion_divergence"] = True
    metrics["vol_cohesion_macro_ratio"] = macro_vol
    metrics["vol_cohesion_micro_ratio"] = micro_vol
    return True
