"""Amortecimento dinamico de Kelly por proximidade da meta de stop win."""

TARGET_DAMPING_FLOOR = 0.40
TARGET_DAMPING_SPAN = 0.60


def resolve_target_proximity_damping(target_win: float, session_pnl: float) -> float:
    """Retorna fator linear entre 0.40 e 1.0 conforme lucro acumulado vs meta."""
    if target_win <= 0.0:
        return 1.0
    remaining_target_pct = max(0.0, (float(target_win) - float(session_pnl)) / float(target_win))
    return TARGET_DAMPING_FLOOR + TARGET_DAMPING_SPAN * remaining_target_pct


def apply_target_proximity_damping(
    kelly_stake_raw: float,
    target_win: float,
    session_pnl: float,
) -> float:
    """Comprime stake Kelly bruta pelo amortecimento de proximidade de alvo."""
    damping = resolve_target_proximity_damping(target_win, session_pnl)
    return max(0.0, float(kelly_stake_raw) * damping)
