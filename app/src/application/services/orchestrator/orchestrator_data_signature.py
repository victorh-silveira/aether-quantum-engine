"""Assinatura multi-timeframe M1+M15 com fronteira de minuto para invalidacao de cache."""

from __future__ import annotations

import time
from typing import Any


__all__ = ["get_data_state_signature", "m1_boundary_epoch", "resolve_signature_boundary_seconds"]


def _resolve_now(now: float | None) -> float:
    """Resolve instante de referencia para fronteira M1 com fallback ao relogio."""
    if now is not None:
        return float(now)
    return time.time()


def resolve_signature_boundary_seconds(orch: Any) -> int:
    """Le fronteira de assinatura em segundos a partir da configuracao do orquestrador."""
    config = getattr(orch, "config", None)
    if not isinstance(config, dict):
        return 60
    orchestrator = config.get("orchestrator")
    if not isinstance(orchestrator, dict):
        return 60
    raw = orchestrator.get("signature_boundary_seconds", 60)
    try:
        return max(60, int(raw))
    except (TypeError, ValueError):
        return 60


def m1_boundary_epoch(orch: Any, *, now: float | None = None) -> int:
    """Retorna epoch Unix alinhado ao minuto corrente do relogio assincrono."""
    now_ts = _resolve_now(now)
    clock_boundary = int(now_ts // 60) * 60
    anchor_epoch = int(getattr(orch, "_last_epoch", 0) or 0)
    if anchor_epoch > 0:
        anchor_boundary = int(anchor_epoch // 60) * 60
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
