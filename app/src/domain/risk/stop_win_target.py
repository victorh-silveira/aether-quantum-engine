"""Alvo absoluto de lucro de sessao conforme faixa de banca inicial."""

import math
from typing import Any


def resolve_stop_win_target(risk_management: dict[str, Any], initial_bankroll: float) -> float:
    """Banca inicial abaixo do limiar usa valor fixo; caso contrario porcentagem da banca inicial."""
    rm = risk_management or {}
    ini = float(initial_bankroll)
    thr = float(rm.get("small_account_threshold", 100.0))
    if ini < thr:
        return max(0.0, float(rm.get("small_account_stop_win", 10.0)))
    pct = float(rm.get("large_account_stop_win_pct", 1.0))
    pct = max(0.0, min(100.0, pct))
    return math.floor(ini * pct / 100.0 * 100) / 100


def resolve_max_stake_pct(
    kelly_config: dict[str, Any],
    conviction: float,
) -> float:
    """Retorna o teto percentual da banca conforme conviccao."""
    base = float(kelly_config.get("max_stake_pct", 0.01))
    threshold = float(kelly_config.get("high_conviction_stake_threshold", 0.75))
    if conviction >= threshold:
        return float(kelly_config.get("max_stake_pct_high_conviction", base))
    return base
