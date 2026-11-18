"""Captura settlement confirmado sem afetar conciliacao financeira."""

from __future__ import annotations

import logging
from typing import Any

from src.infrastructure.market.contract_audit import settlement_audit_row


logger = logging.getLogger("AETH")


async def record_settlement_audit(
    orch,
    payload: dict,
    contract,
    symbol: str,
    *,
    signal_prob: float | None = None,
) -> None:
    """Persistencia best-effort; falha de auditoria nao reabre a ordem."""
    writer = getattr(orch, "market_writer", None)
    if writer is None or not hasattr(writer, "enqueue_contract_audit"):
        return
    direction = getattr(contract, "direction", None)
    direction_name = str(getattr(direction, "value", direction) or "UNKNOWN")
    mode = str(getattr(orch, "config", {}).get("trading", {}).get("mode", "demo"))
    try:
        row = settlement_audit_row(
            payload,
            symbol=str(symbol),
            mode=mode,
            direction=direction_name,
            signal_prob=signal_prob,
        )
        await writer.enqueue_contract_audit(row)
    except Exception as exc:
        logger.error("AUDIT: settlement sem captura cid=%s erro=%s", payload.get("contract_id"), exc)


def is_transient_broker_error(exc: BaseException) -> bool:
    """Indica falha temporaria de rede ou broker."""
    name = type(exc).__name__
    if name in {"ConnectionClosed", "ConnectionClosedError", "ConnectionClosedOK"}:
        return True
    return isinstance(exc, (TimeoutError, ConnectionError, OSError))


def mark_ws_offline(ws: Any) -> None:
    """Marca WebSocket como desconectado para forcar reconexao."""
    if ws is not None:
        ws.is_running = False


def min_elapsed_before_stagnant_polls(risk_params: dict | None, execution_cfg: dict | None) -> float:
    """Segundos minimos antes de contar polls estagnados (fallback estatico)."""
    ex = execution_cfg or {}
    if ex.get("settlement_stagnant_grace_seconds") is not None:
        return max(0.0, float(ex["settlement_stagnant_grace_seconds"]))
    slack = float(ex.get("settlement_post_expiry_slack_seconds", 25.0))
    p = risk_params or {}
    dur_val = p.get("duration", 1)
    if dur_val == "MULT":
        return 3600.0
    try:
        dur = max(1, int(dur_val))
    except (ValueError, TypeError):
        dur = 1
    unit = str(p.get("duration_unit", "m")).lower().strip()
    if unit == "m":
        return float(dur * 60 + slack)
    if unit == "t":
        est = float(ex.get("settlement_tick_seconds_estimate", 2.5))
        return float(dur * est + slack)
    if unit == "s":
        return float(dur + slack)
    return float(dur * 60 + slack)


def calculate_cluster_grace_period(active_contracts: dict, execution_cfg: dict, start_time: float) -> float:
    """Calcula o tempo de carencia dinamico baseado na expiracao mais longa do cluster."""
    ex = execution_cfg or {}
    slack = float(ex.get("settlement_post_expiry_slack_seconds", 45.0))
    if not active_contracts:
        return 0.0
    max_expiry = 0
    for contract in active_contracts.values():
        if hasattr(contract, "expiry_time") and contract.expiry_time:
            max_expiry = max(max_expiry, int(contract.expiry_time))
    if max_expiry > 0:
        diff = max_expiry - int(start_time)
        return float(max(diff, 0) + slack)
    return 0.0


def prune_orphan_contract_ids(active_ids: list[int], active_contracts: dict) -> tuple[list[int], list[int]]:
    """Retorna lista saneada de IDs ativos e IDs orfaos."""
    state_ids = {int(c_id) for c_id in active_contracts}
    kept = [c_id for c_id in active_ids if int(c_id) in state_ids]
    orphan = [c_id for c_id in active_ids if int(c_id) not in state_ids]
    return kept, orphan


def clear_contract_tracking(ids: list[int], risk_manager: Any) -> None:
    """Limpa rastreamento de contratos ativos no gerenciador de risco."""
    risk_manager.active_contract_ids = []
    clear_contract_metadata(ids, risk_manager)


def clear_contract_metadata(ids: list[int], risk_manager: Any) -> None:
    """Remove vinculos auxiliares de contratos no gerenciador de risco."""
    for c_id in ids:
        risk_manager.contract_to_symbol.pop(int(c_id), None)
        risk_manager.cluster_results.pop(int(c_id), None)
