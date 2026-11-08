"""Auditoria de compras confirmadas sem repetir a ordem."""

from __future__ import annotations

from typing import Any

from src.infrastructure.market.contract_audit import open_audit_row


async def record_purchase_audit(
    writer: Any,
    logger: Any,
    config: dict,
    payload: dict,
    symbol: str,
    direction: str,
    request_ms: int,
    ack_ms: int,
) -> None:
    """Persiste a compra; falha de captura nao produz nova compra."""
    if writer is None:
        return
    try:
        row = open_audit_row(
            payload,
            symbol=symbol,
            mode=str(config.get("trading", {}).get("mode", "demo")),
            direction=direction,
            request_epoch_ms=request_ms,
            ack_epoch_ms=ack_ms,
        )
        await writer.enqueue_contract_audit(row)
    except Exception as exc:
        logger.error("AUDIT: compra confirmada sem captura cid=%s erro=%s", payload.get("contract_id"), exc)
