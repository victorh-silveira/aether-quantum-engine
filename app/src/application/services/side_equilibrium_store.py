"""Store rolling CALL/PUT outcomes (memoria + Redis + Timescale)."""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any

from src.domain.analytics.side_equilibrium import SideCounts, SideEquilibriumConfig, parse_side_equilibrium_config


logger = logging.getLogger("AETH")

_REDIS_KEY_TMPL = "stats:side:{symbol}"


def _ensure_bag(orch: Any) -> dict[str, deque]:
    """Garante o dicionario rolling de historico no orch."""
    bag = getattr(orch, "_side_equilibrium_hist", None)
    if bag is None:
        orch._side_equilibrium_hist = {}
        bag = orch._side_equilibrium_hist
    return bag


def side_eq_config_from_orch(orch: Any) -> SideEquilibriumConfig:
    """Le configuracao side_equilibrium do orch."""
    exec_cfg = {}
    config = getattr(orch, "config", None)
    if isinstance(config, dict):
        orch_cfg = config.get("orchestrator") if isinstance(config.get("orchestrator"), dict) else {}
        exec_cfg = orch_cfg.get("execution") if isinstance(orch_cfg.get("execution"), dict) else {}
    raw = exec_cfg.get("side_equilibrium") if isinstance(exec_cfg, dict) else None
    return parse_side_equilibrium_config(raw if isinstance(raw, dict) else {})


def counts_from_hist(hist: deque, *, window: int) -> SideCounts:
    """Agrega contagens CALL/PUT a partir do historico."""
    rows = list(hist)[-max(1, int(window)) :]
    call_n = call_wins = put_n = put_wins = 0
    for direction, won in rows:
        if direction == "CALL":
            call_n += 1
            if won:
                call_wins += 1
        elif direction == "PUT":
            put_n += 1
            if won:
                put_wins += 1
    return SideCounts(call_n=call_n, call_wins=call_wins, put_n=put_n, put_wins=put_wins)


def snapshot_side_counts(orch: Any, symbol: str, *, window: int) -> SideCounts:
    """Snapshot das contagens do simbolo na janela."""
    bag = _ensure_bag(orch)
    hist = bag.get(str(symbol))
    if hist is None:
        return SideCounts()
    return counts_from_hist(hist, window=window)


def record_side_equilibrium_outcome(
    orch: Any,
    symbol: str,
    *,
    direction: str | None,
    won: bool,
    profit: float = 0.0,
    raw_prob: float | None = None,
    calibrated_prob: float | None = None,
    cycle_id: int | None = None,
) -> SideCounts:
    """Registra outcome CALL/PUT e persiste best-effort."""
    dir_name = str(direction or "").upper()
    if dir_name not in {"CALL", "PUT"}:
        return snapshot_side_counts(orch, str(symbol), window=side_eq_config_from_orch(orch).large_window)
    cfg = side_eq_config_from_orch(orch)
    bag = _ensure_bag(orch)
    sym = str(symbol)
    hist = bag.get(sym)
    if hist is None:
        hist = deque(maxlen=max(cfg.large_window, cfg.small_window))
        bag[sym] = hist
    hist.append((dir_name, bool(won)))
    counts = counts_from_hist(hist, window=cfg.large_window)
    _persist_redis_best_effort(orch, sym, counts)
    _persist_timescale_best_effort(
        orch,
        symbol=sym,
        direction=dir_name,
        won=bool(won),
        profit=float(profit),
        raw_prob=raw_prob,
        calibrated_prob=calibrated_prob,
        cycle_id=cycle_id,
    )
    return counts


def _persist_redis_best_effort(orch: Any, symbol: str, counts: SideCounts) -> None:
    """Persiste contagens no Redis quando disponivel."""
    store = getattr(orch, "state_store", None) or getattr(orch, "redis_store", None)
    client = getattr(store, "client", None) if store is not None else None
    if client is None:
        return
    key = _REDIS_KEY_TMPL.format(symbol=symbol)
    mapping = {
        "call_n": str(counts.call_n),
        "call_wins": str(counts.call_wins),
        "put_n": str(counts.put_n),
        "put_wins": str(counts.put_wins),
        "updated_at": str(int(time.time())),
    }
    try:
        hset = getattr(client, "hset", None)
        if hset is None:
            return
        result = hset(key, mapping=mapping)
        if hasattr(result, "__await__"):
            return
    except Exception as exc:
        logger.debug("SIDE_EQ redis skip: %s", exc)


def _persist_timescale_best_effort(
    orch: Any,
    *,
    symbol: str,
    direction: str,
    won: bool,
    profit: float,
    raw_prob: float | None,
    calibrated_prob: float | None,
    cycle_id: int | None,
) -> None:
    """Enfileira outcome no Timescale quando disponivel."""
    writer = getattr(orch, "timescale_writer", None) or getattr(orch, "_timescale_writer", None)
    if writer is None:
        infra = getattr(orch, "infra", None)
        writer = getattr(infra, "timescale_writer", None) if infra is not None else None
    enqueue = getattr(writer, "enqueue_trade_outcome", None) if writer is not None else None
    if enqueue is None:
        return
    try:
        result = enqueue(
            symbol=symbol,
            direction=direction,
            won=won,
            profit=profit,
            raw_prob=raw_prob,
            calibrated_prob=calibrated_prob,
            cycle_id=cycle_id,
        )
        if hasattr(result, "__await__"):
            loop_create = getattr(orch, "create_task", None)
            if callable(loop_create):
                loop_create(result)
    except Exception as exc:
        logger.debug("SIDE_EQ timescale skip: %s", exc)
