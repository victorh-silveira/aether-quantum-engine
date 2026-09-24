"""Normaliza somente campos de execucao fornecidos pela corretora."""

from __future__ import annotations

from typing import Any


def _number(payload: dict[str, Any], *keys: str) -> float | None:
    """Extrai o primeiro numero presente sem substituir ausencia por zero."""
    for key in keys:
        value = payload.get(key)
        if value is not None and value != "":
            return float(value)
    return None


def _integer(payload: dict[str, Any], *keys: str) -> int | None:
    """Converte epoch opcional sem criar timestamp estimado."""
    value = _number(payload, *keys)
    return int(value) if value is not None else None


def open_audit_row(
    payload: dict[str, Any],
    *,
    symbol: str,
    mode: str,
    direction: str,
    request_epoch_ms: int,
    ack_epoch_ms: int,
) -> dict[str, Any]:
    """Prepara compra confirmada; nao usa spot de proposta como spot de entrada."""
    return {
        "contract_id": int(payload["contract_id"]),
        "symbol": symbol,
        "account_mode": mode,
        "direction": direction,
        "transaction_buy_id": str(payload["transaction_id"]) if payload.get("transaction_id") is not None else None,
        "request_epoch_ms": int(request_epoch_ms),
        "ack_epoch_ms": int(ack_epoch_ms),
        "date_start": _integer(payload, "start_time", "date_start"),
        "date_expiry": _integer(payload, "date_expiry"),
        "buy_price": _number(payload, "buy_price"),
        "payout": _number(payload, "payout"),
        "settlement_source": "pending",
    }


def settlement_audit_row(
    payload: dict[str, Any],
    *,
    symbol: str,
    mode: str,
    direction: str,
    signal_prob: float | None = None,
) -> dict[str, Any]:
    """Prepara settlement; fonte sintetica nunca vira observacao broker."""
    source = str(payload.get("audit_source") or "broker")
    if source not in {"broker", "inferred_rest", "profit_table"}:
        source = "unknown"
    return {
        "contract_id": int(payload["contract_id"]),
        "symbol": symbol,
        "account_mode": mode,
        "direction": direction,
        "date_start": _integer(payload, "date_start"),
        "date_expiry": _integer(payload, "date_expiry"),
        "entry_tick": _number(payload, "entry_spot", "entry_tick"),
        "entry_tick_time": _integer(payload, "entry_spot_time", "entry_tick_time"),
        "exit_tick": _number(payload, "exit_spot", "exit_tick"),
        "exit_tick_time": _integer(payload, "exit_spot_time", "exit_tick_time"),
        "buy_price": _number(payload, "buy_price"),
        "payout": _number(payload, "payout"),
        "signal_prob": float(signal_prob) if signal_prob is not None else None,
        "profit": _number(payload, "profit"),
        "status": str(payload["status"]) if payload.get("status") is not None else None,
        "settlement_source": source,
    }
