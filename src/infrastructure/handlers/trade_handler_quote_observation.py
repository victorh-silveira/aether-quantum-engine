"""Registro auditavel da cotacao antes da decisao de compra."""

from __future__ import annotations

from typing import Any


async def record_indicator_quote(
    writer: Any,
    config: dict,
    symbol: str,
    direction: str,
    parameters: dict,
    proposal_id: str,
    price: float,
    payout: float | None,
    payout_rate: float | None,
) -> None:
    """Registra cotacao observada no Timescale quando habilitado."""
    _ = (writer, config, symbol, direction, parameters, proposal_id, price, payout, payout_rate)
