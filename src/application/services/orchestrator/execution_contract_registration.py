"""Registro atomico de compra confirmada frente a liquidacoes rapidas."""

from typing import Any

from src.application.services.orchestrator.settlement_dedupe import settlement_lock, settlement_seen


async def register_confirmed_contract(orch: Any, contract: Any, symbol: str, stake: float) -> bool:
    """Rastreia uma compra uma vez antes de iniciar sua inscricao de liquidacao."""
    contract_id = int(contract.contract_id)
    async with settlement_lock(orch):
        if contract_id in settlement_seen(orch):
            return False
        if contract_id in getattr(orch.state, "active_contracts", {}):
            return True
        rm = orch.risk_manager
        symbols = getattr(rm, "contract_to_symbol", None)
        if not isinstance(symbols, dict):
            symbols = {}
            rm.contract_to_symbol = symbols
        symbols[contract_id] = str(symbol)
        rm.record_contract_stake(contract_id, float(stake))
        if contract_id not in rm.active_contract_ids:
            rm.active_contract_ids.append(contract_id)
        await orch.state.add_contract(contract)
        orch._contract_cycle[contract_id] = int(orch._active_cycle_id)
        return True
