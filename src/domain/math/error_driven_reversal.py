"""Calculo matematico de Brier Score individual e gating para Alpha Flip Adaptativo."""

from __future__ import annotations


def calculate_trade_brier_score(
    prob: float,
    *,
    won: bool,
    direction: str,
) -> tuple[float, float]:
    """Calcula o Brier score do evento e o residuo direcional epsilon = y - p."""
    p_clamped = max(0.0, min(1.0, float(prob)))
    dir_upper = str(direction or "").upper().strip()
    if dir_upper == "CALL":
        y = 1.0 if won else 0.0
    elif dir_upper == "PUT":
        y = 0.0 if won else 1.0
    else:
        y = 1.0 if won else 0.0
    epsilon = float(y - p_clamped)
    brier = float(epsilon**2)
    return brier, epsilon


def is_error_driven_reversal_armed(
    brier_score: float,
    prob: float,
    *,
    threshold: float = 0.40,
    min_conviction: float = 0.60,
) -> bool:
    """Indica se um trade qualifica para reversao direcional adaptativa no ciclo seguinte."""
    bs = float(brier_score)
    p_clamped = max(0.0, min(1.0, float(prob)))
    conviction = max(p_clamped, 1.0 - p_clamped)
    if bs < float(threshold):
        return False
    return conviction >= float(min_conviction)


def resolve_target_reversal_direction(direction: str) -> str:
    """Inverte a direcao de mercado para fluxo de contra-ataque adaptativo."""
    dir_clean = str(direction or "").upper().strip()
    if dir_clean == "CALL":
        return "PUT"
    if dir_clean == "PUT":
        return "CALL"
    return "PUT"
