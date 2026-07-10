"""Assinatura multi-timeframe M1+M15 com fronteira de minuto para invalidacao de cache."""

from __future__ import annotations

import time
from typing import Any


__all__ = [
    "at_signature_boundary",
    "get_data_state_signature",
    "m1_boundary_epoch",
    "resolve_signature_boundary_seconds",
    "seconds_until_next_signature_boundary",
]


def _resolve_now(now: float | None) -> float:
    """Resolve instante de referencia para fronteira temporal com fallback ao relogio."""
    if now is not None:
        return float(now)
    return time.time()


def _orchestrator_cfg(orch: Any) -> dict:
    """Retorna bloco orchestrator da configuracao quando disponivel."""
    config = getattr(orch, "config", None)
    if not isinstance(config, dict):
        return {}
    chunk = config.get("orchestrator")
    return chunk if isinstance(chunk, dict) else {}


def resolve_signature_boundary_seconds(orch: Any) -> int:
    """Le fronteira de assinatura em segundos a partir da configuracao do orquestrador."""
    orchestrator = _orchestrator_cfg(orch)
    if not orchestrator:
        return 60
    raw = orchestrator.get("signature_boundary_seconds")
    if raw is not None:
        try:
            return max(60, int(raw))
        except (TypeError, ValueError):
            pass
    cadence = orchestrator.get("cycle_interval_seconds")
    if cadence is not None:
        try:
            cadence_int = int(cadence)
            if cadence_int >= 60:
                return cadence_int
        except (TypeError, ValueError):
            pass
    return 60


def seconds_until_next_signature_boundary(orch: Any, *, now: float | None = None) -> float:
    """Calcula segundos restantes ate a proxima fronteira temporal configurada."""
    boundary = resolve_signature_boundary_seconds(orch)
    now_ts = _resolve_now(now)
    next_boundary = (int(now_ts) // boundary + 1) * boundary
    return max(0.0, float(next_boundary) - now_ts)


def at_signature_boundary(orch: Any, *, now: float | None = None, tolerance: float = 1.0) -> bool:
    """True quando o instante corrente esta sobre a fronteira temporal configurada."""
    boundary = resolve_signature_boundary_seconds(orch)
    now_ts = _resolve_now(now)
    epoch = int(now_ts)
    offset = epoch % boundary
    tol = max(0.0, float(tolerance))
    return offset <= tol or offset >= max(0, boundary - tol)


def m1_boundary_epoch(orch: Any, *, now: float | None = None) -> int:
    """Retorna epoch Unix alinhado a fronteira temporal operacional configurada."""
    boundary = resolve_signature_boundary_seconds(orch)
    now_ts = _resolve_now(now)
    clock_boundary = int(now_ts // boundary) * boundary
    anchor_epoch = int(getattr(orch, "_last_epoch", 0) or 0)
    if anchor_epoch > 0:
        anchor_boundary = int(anchor_epoch // boundary) * boundary
        return max(clock_boundary, anchor_boundary)
    return clock_boundary


def _last_candle_epoch(candles: list) -> int | None:
    """Extrai epoch do ultimo candle fechado em uma serie OHLC."""
    if not candles:
        return None
    return int(candles[-1].epoch)


def get_data_state_signature(orch: Any, *, now: float | None = None) -> str:
    """Monta assinatura M1+M15 com fronteira M1 obrigatoria por minuto."""
    stream = getattr(orch, "stream", None)
    if stream is None:
        return ""
    boundary = m1_boundary_epoch(orch, now=now)
    micro_parts: list[str] = []
    macro_parts: list[str] = []
    for sym in getattr(orch, "symbols", []):
        micro_hist = getattr(stream, "micro_candles", {}).get(sym, [])
        micro_epoch = _last_candle_epoch(micro_hist)
        if micro_epoch is not None:
            micro_parts.append(f"{sym}@{micro_epoch}")
        macro_store = getattr(stream, "macro_candles", None)
        if macro_store is None:
            macro_store = getattr(stream, "candles", {})
        macro_hist = macro_store.get(sym, []) if isinstance(macro_store, dict) else []
        macro_epoch = _last_candle_epoch(macro_hist)
        if macro_epoch is not None:
            macro_parts.append(f"{sym}@{macro_epoch}")
    micro_sig = "|".join(micro_parts)
    macro_sig = "|".join(macro_parts)
    if not micro_sig and not macro_sig:
        return ""
    return f"m1b:{boundary};m1:{micro_sig};m15:{macro_sig}"
