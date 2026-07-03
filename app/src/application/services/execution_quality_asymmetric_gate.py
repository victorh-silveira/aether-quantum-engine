"""Veto assimétrico de conviccao em regime neutro macro M15."""

from __future__ import annotations

from src.domain.risk.stake_sizing import raw_side_from_metrics


RECOVERY_ASYMMETRIC_MIN_SCORE = 0.68
NEUTRAL_REGIME_TOKEN = "NEUTRO"


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
    if not _is_neutral_universal_regime(metrics):
        return False
    score = _effective_signal(metrics)
    if score + 1e-9 >= float(min_conviction_floor):
        return False
    metrics["regime_skip_cycle"] = True
    metrics["gate_reason"] = "low_conviction_neutral_skip"
    metrics["recovery_asymmetric_gate"] = True
    return True
