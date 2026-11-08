"""Captura settlement confirmado sem afetar conciliacao financeira."""

from __future__ import annotations

import logging

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
