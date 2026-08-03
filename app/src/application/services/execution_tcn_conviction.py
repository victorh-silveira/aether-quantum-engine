"""Heuristicas de conviccao TCN para travar direcao e waivers de gate."""

from __future__ import annotations

import contextlib
from typing import Any


_TCN_LOCK_MARGIN = 0.12
_TCN_HIGH_CONVICTION_MARGIN = 0.25


def tcn_direction_margin(metrics: dict[str, Any] | None) -> float:
    """Margem direcional efetiva a partir de direction_margin ou calibrated_prob."""
    bag = metrics if isinstance(metrics, dict) else {}
    margin = 0.0
    raw_margin = bag.get("direction_margin")
    if raw_margin is not None:
        with contextlib.suppress(TypeError, ValueError):
            margin = abs(float(raw_margin))
    cal = bag.get("calibrated_prob")
    if cal is None:
        cal = bag.get("raw_prob")
    if cal is not None:
        with contextlib.suppress(TypeError, ValueError):
            margin = max(margin, abs(float(cal) - 0.5))
    return float(margin)


def tcn_direction_lock_active(metrics: dict[str, Any] | None) -> bool:
    """True quando a margem TCN e suficiente para bloquear flip de lado."""
    return tcn_direction_margin(metrics) + 1e-12 >= _TCN_LOCK_MARGIN


def tcn_high_conviction_active(metrics: dict[str, Any] | None) -> bool:
    """True quando a margem TCN justifica waiver de gates duros."""
    return tcn_direction_margin(metrics) + 1e-12 >= _TCN_HIGH_CONVICTION_MARGIN
