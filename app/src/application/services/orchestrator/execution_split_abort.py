"""Tratativas anti-loop para abortos tecnicos em fatiamento de ordens."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from src.domain.models.trade import TradeDirection


def next_split_attempt_seq(orch: Any) -> int:
    """Incrementa e retorna a sequencia global de tentativas de fatiamento da sessao."""
    seq = int(getattr(orch, "_split_attempt_seq", 0) or 0) + 1
    orch._split_attempt_seq = seq
    orch._last_split_attempt_seq = seq
    return seq


async def handle_split_abort(
    orch: Any,
    logger: Any,
    *,
    symbol: str,
    direction: TradeDirection,
    cycle_id: int,
) -> None:
    """Registra bloqueio anti-loop apos abort de fatiamento e cede cooperativamente o loop."""
    signature = str(orch.get_data_state_signature() or "") if hasattr(orch, "get_data_state_signature") else ""
    orch._last_split_abort_signature, orch._last_split_abort_minute = signature, int(time.time() // 60)
    orch._last_split_abort_symbol, orch._last_split_abort_direction = str(symbol), str(direction.name)
    orch._last_split_abort_cycle_id = cycle_id
    orch.is_trading = False
    split_attempt_seq = int(getattr(orch, "_last_split_attempt_seq", 0) or 0)
    logger.warning(
        "[C%04d] EXEC_SPLIT_ABORT | %s %s | lote fracionado abortado | seq=%d | assinatura=%s",
        cycle_id,
        symbol,
        direction.name,
        split_attempt_seq,
        signature or "-",
    )
    await asyncio.sleep(0)
