"""Veto assimétrico de conviccao em regime neutro macro M15."""

from __future__ import annotations

from src.application.services.execution_universal_regime_types import MICRO_MIDDLE_UNCERTAINTY_REASON
from src.domain.risk.stake_sizing import raw_side_from_metrics


RECOVERY_ASYMMETRIC_MIN_SCORE = 0.68
NEUTRAL_REGIME_TOKEN = "NEUTRO"
MICRO_ADX_FLOOR = 0.15
MICRO_BB_EXTREME = 0.01
MICRO_HURST_RANDOM_WALK_MAX = 0.48


def _effective_signal(metrics: dict) -> float:
    """Retorna trade_score calibrado com fallback para conviccao bruta."""
    score = float(metrics.get("trade_score", metrics.get("conviction", 0.0)))
    return max(score, raw_side_from_metrics(metrics))


def _is_neutral_universal_regime(metrics: dict) -> bool:
    """True quando o barramento macro nao classificou tendencia, compressao ou exaustao."""
    regime = metrics.get("universal_regime") or metrics.get("universal_regime_scenario")
    if regime is None:
        return True
    token = str(regime).strip().upper()
    return token in ("", NEUTRAL_REGIME_TOKEN, "NEUTRAL")


def validate_recovery_asymmetric_gate(
    metrics: dict,
    *,
    min_conviction_floor: float = RECOVERY_ASYMMETRIC_MIN_SCORE,
) -> bool:
    """Veto absoluto de ciclo em regime neutro com conviccao abaixo do piso institucional."""
    if str(metrics.get("gate_reason") or "") == MICRO_MIDDLE_UNCERTAINTY_REASON:
        return True
    if not _is_neutral_universal_regime(metrics):
        return False
    score = _effective_signal(metrics)
    if score + 1e-9 >= float(min_conviction_floor):
        return False
    metrics["regime_skip_cycle"] = True
    metrics["gate_reason"] = "low_conviction_neutral_skip"
    metrics["recovery_asymmetric_gate"] = True
    return True


def _micro_indicators(metrics: dict) -> tuple[float, float, float]:
    """Extrai ADX, bb_width e Hurst do bloco micro M1 com fallbacks neutros."""
    indicators = metrics.get("indicators") or {}
    adx = float(indicators.get("adx", 1.0))
    bb_width = float(indicators.get("bb_width", 1.0))
    hurst = float(indicators.get("hurst", 0.5))
    return adx, bb_width, hurst


def _micro_noise_gate_params(exec_cfg: dict | None) -> tuple[bool, float, float, float]:
    """Le pisos do micro noise gate da config de execucao com fallback institucional."""
    chunk = exec_cfg.get("micro_noise_gate") if isinstance(exec_cfg, dict) else None
    chunk = chunk if isinstance(chunk, dict) else {}
    enabled = bool(chunk.get("enabled", True))
    adx_floor = float(chunk.get("adx_floor", MICRO_ADX_FLOOR))
    bb_extreme = float(chunk.get("bb_extreme", MICRO_BB_EXTREME))
    hurst_max = float(chunk.get("hurst_random_walk_max", MICRO_HURST_RANDOM_WALK_MAX))
    return enabled, adx_floor, bb_extreme, hurst_max


def validate_micro_noise_gate(metrics: dict, *, exec_cfg: dict | None = None) -> bool:
    """Veto de ciclo em chop micro: ADX colapsado ou squeeze em random walk sem reversao."""
    enabled, adx_floor, bb_extreme, hurst_max = _micro_noise_gate_params(exec_cfg)
    if not enabled:
        return False
    adx, bb_width, hurst = _micro_indicators(metrics)
    if adx + 1e-9 < adx_floor:
        metrics["regime_skip_cycle"] = True
        metrics["gate_reason"] = "micro_adx_chop_skip"
        metrics["micro_noise_gate"] = True
        return True
    if bb_width + 1e-9 < bb_extreme and hurst + 1e-9 < hurst_max:
        metrics["regime_skip_cycle"] = True
        metrics["gate_reason"] = "micro_squeeze_breakout_skip"
        metrics["micro_noise_gate"] = True
        return True
    return False


def validate_micro_boundary_saturation_gate(metrics: dict) -> bool:
    """Veto absoluto quando o filtro de exaustao micro rebaixou o score por saturacao de banda."""
    if not metrics.get("micro_boundary_exhaustion"):
        return False
    metrics["regime_skip_cycle"] = True
    metrics["gate_reason"] = "micro_boundary_saturation_skip"
    metrics["micro_boundary_saturation_gate"] = True
    return True
