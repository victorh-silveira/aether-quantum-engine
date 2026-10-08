"""Atribuicao de direcao pelo spot confirmado do contrato Rise/Fall."""

from __future__ import annotations

import math


def broker_price_audit(payload: dict, direction: str | None) -> str | None:
    """Resume deslocamento real sem inferir outcome a partir da vela M5."""
    if str(payload.get("audit_source") or "broker") != "broker":
        return None
    raw_entry = payload.get("entry_spot", payload.get("entry_tick"))
    raw_exit = payload.get("exit_spot", payload.get("exit_tick"))
    try:
        entry = float(raw_entry)
        exit_ = float(raw_exit)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) and value > 0.0 for value in (entry, exit_)):
        return None
    side = str(direction or "").upper()
    if side not in {"CALL", "PUT"}:
        return None
    delta = exit_ - entry
    observed = "CALL" if delta > 0.0 else "PUT" if delta < 0.0 else "FLAT"
    phase = ""
    raw_time = payload.get("entry_spot_time", payload.get("entry_tick_time"))
    try:
        entry_time = int(raw_time)
    except (TypeError, ValueError):
        entry_time = 0
    if entry_time > 0:
        phase = f" | FASE_M5: {entry_time % 300}s"
    return f"SPOT: {entry:.2f}->{exit_:.2f} ({delta:+.2f}) | REAL: {observed}{phase}"
