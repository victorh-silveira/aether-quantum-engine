"""Calculo e aplicacao de quarentena de perda alinhada a velas M15."""

from __future__ import annotations

import math
import time
from typing import Any


def resolve_loss_symbol(resolved_meta: Any, contract: Any, orch: Any, c_id: Any) -> str | None:
    """Extrai o simbolo do contrato liquidado."""
    sym = (
        (resolved_meta or {}).get("symbol")
        if isinstance(resolved_meta, dict)
        else getattr(contract, "symbol", None) or getattr(orch.risk_manager, "contract_to_symbol", {}).get(c_id)
    )
    return str(sym) if sym else None


def resolve_loss_direction(resolved_meta: Any, contract: Any) -> str | None:
    """Extrai a direcao do contrato liquidado."""
    dir_val = (
        (resolved_meta or {}).get("direction") or (resolved_meta or {}).get("indicator_side")
        if isinstance(resolved_meta, dict)
        else None
    )
    if not dir_val:
        dir_obj = getattr(contract, "direction", None)
        dir_val = getattr(dir_obj, "name", None) if dir_obj else None
    return str(dir_val).upper() if dir_val else None


def calculate_loss_cooldown_expiry(
    orch: Any,
    symbol: str,
    now_ts: float,
) -> tuple[float, int]:
    """Calcula expiracao da quarentena alinhada com o final do ciclo M15."""
    pb_cfg = orch.config.get("pullback_strategy", {}) if isinstance(orch.config, dict) else {}
    if not bool(pb_cfg.get("enabled", False)):
        return now_ts + 900.0, 900
    period_seconds = int(pb_cfg.get("period_seconds", 900))
    cooldown_candles = int(pb_cfg.get("loss_cooldown_candles", 1))
    consec_cooldown = int(pb_cfg.get("consecutive_loss_cooldown_candles", 2))
    consecutive = int(getattr(orch, "_symbol_consecutive_losses", {}).get(symbol, 1))
    candles = consec_cooldown if consecutive >= 2 else cooldown_candles
    if candles <= 0:
        return 0.0, 0
    boundary = math.ceil(now_ts / period_seconds) * period_seconds
    if boundary <= now_ts:
        boundary += period_seconds
    expiry = boundary + (candles - 1) * period_seconds - 5.0
    expiry_ts = max(now_ts, expiry)
    ttl = max(1, int(math.ceil(expiry_ts - now_ts)))
    return expiry_ts, ttl


async def handle_settlement_loss_quarantine(
    orch: Any,
    resolved_meta: Any,
    contract: Any,
    c_id: Any,
    profit: float,
) -> None:
    """Aplica quarentena estrutural e atualiza perdas consecutivas por simbolo."""
    symbol = resolve_loss_symbol(resolved_meta, contract, orch, c_id)
    if not symbol:
        return
    tracker = getattr(orch, "_session_pair_tracker", None)
    if tracker is not None and hasattr(tracker, "record_symbol_settlement"):
        tracker.record_symbol_settlement(symbol, won=(profit >= 0.0), profit=profit)
    consec_map = orch.__dict__.setdefault("_symbol_consecutive_losses", {})
    reversal_map = orch.__dict__.setdefault("_pending_loss_reversals", {})
    pb_cfg = orch.config.get("pullback_strategy", {}) if isinstance(orch.config, dict) else {}
    reversal_on_loss = bool(pb_cfg.get("reversal_on_loss", False))
    if profit >= 0.0:
        consec_map[symbol] = 0
        reversal_map.pop(symbol, None)
        return
    consec_map[symbol] = int(consec_map.get(symbol, 0)) + 1
    dir_lost = resolve_loss_direction(resolved_meta, contract)
    if dir_lost in ("CALL", "PUT") and reversal_on_loss:
        target_opp = "PUT" if dir_lost == "CALL" else "CALL"
        reversal_map[symbol] = {"from_dir": dir_lost, "target_dir": target_opp, "armed_at": time.time()}
        orch.logger.info("ALPHA_REVERSAL | %s armado para %s apos LOSS em %s", symbol, target_opp, dir_lost)
    now_ts = time.time()
    quarantine_ts, ttl = calculate_loss_cooldown_expiry(orch, symbol, now_ts)
    if quarantine_ts > 0.0:
        orch.__dict__.setdefault("_symbol_loss_quarantine", {})[symbol] = quarantine_ts
        r_store = getattr(orch, "state_store", None)
        if r_store is not None and hasattr(r_store, "redis") and r_store.redis is not None:
            try:
                mode = str(orch.config.get("trading", {}).get("mode", "demo")).lower()
                await r_store.redis.set(f"quarantine:loss:{mode}:{symbol}", "1", ex=ttl)
            except Exception as exc:
                orch.logger.debug("Falha ao registrar quarentena no Redis para %s: %s", symbol, exc)
