"""Pre-condicoes e aquisicao de lock para iniciar um ciclo de trade."""

from __future__ import annotations

from typing import Any


def trading_cycle_entry_allowed(orch: Any) -> bool:
    """False quando o motor nao pode iniciar um novo ciclo de decisao."""
    if orch.is_trading:
        return False
    if orch.state.active_contracts:
        if not orch._settlement_wait_logged:
            orch.logger.info(
                "CICLO: aguardando liquidacao (%d contrato(s) aberto(s))",
                len(orch.state.active_contracts),
            )
            orch._settlement_wait_logged = True
        return False
    orch._settlement_wait_logged = False
    return True


async def acquire_trading_cycle_lock(orch: Any) -> bool:
    """Reserva o slot de ciclo ativo; False se outro ciclo ja esta em andamento."""
    async with orch.lock:
        if orch.is_trading:
            return False
        orch.is_trading = True
    return True
