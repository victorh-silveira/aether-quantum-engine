"""Gerenciamento de estado compartilhado para o Aether Engine."""

import asyncio
from dataclasses import asdict
from typing import Any

from src.domain.models.trade import Contract, TradeDirection, TradeStatus


class TradingState:
    """Um Singleton thread-safe que rastreia o estado compartilhado do motor de trading.

    Gerencia contratos ativos, saldo da conta e sinalizadores de trading exclusivos para
    garantir a integridade sistêmica durante operações simultâneas.
    """

    _instance = None

    def __new__(cls):
        """Garante uma instância singleton do estado do trading."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_state()
        return cls._instance

    def _init_state(self):
        """Inicialização interna do estado."""
        self.lock = asyncio.Lock()
        self.active_contracts: dict[int, Contract] = {}
        self.balance = 0.0
        self.is_trading = False
        self.last_tick: Any = None

    @classmethod
    def reset(cls):
        """Redefine a instância singleton para fins de teste."""
        if cls._instance:
            cls._instance._init_state()

    async def update_last_tick(self, tick: Any):
        """Atualiza as informações mais recentes do tick de mercado.

        Args:
            tick (Any): O modelo de tick mais recente ou pacote de dados.
        """
        async with self.lock:
            self.last_tick = tick

    async def set_trading(self, *, value: bool):
        """Define o status atual do trading.

        Args:
            value (bool): True se o trading estiver ativo, False caso contrário.
        """
        async with self.lock:
            self.is_trading = value

    async def add_contract(self, contract: Contract):
        """Adiciona um contrato recém-comprado ao estado de rastreamento ativo.

        Args:
            contract (Contract): Os detalhes do contrato para rastrear.
        """
        async with self.lock:
            self.active_contracts[contract.contract_id] = contract

    async def finalize_contract(self, contract_id: int) -> Contract | None:
        """Remove e retorna atomicamente um contrato se ele existir no estado ativo.

        Este método é crítico para evitar a contagem dupla de ganhos/perdas
        durante a reconciliação simultânea e atualizações de fluxo.

        Args:
            contract_id (int): O identificador do contrato a ser finalizado.

        Returns:
            Contract | None: O contrato se estivesse ativo, None caso contrário.
        """
        async with self.lock:
            if contract_id in self.active_contracts:
                return self.active_contracts.pop(contract_id)
            return None

    async def get_state(self) -> dict[str, Any]:
        """Retorna o estado atual do trading para persistência.

        Returns:
            Dict[str, Any]: Estado da sessão atual.
        """
        async with self.lock:
            contracts_serialized = {}
            for cid, c in self.active_contracts.items():
                data = asdict(c)
                data["status"] = c.status.name
                data["direction"] = c.direction.name
                contracts_serialized[str(cid)] = data

            return {"balance": self.balance, "is_trading": self.is_trading, "active_contracts": contracts_serialized}

    async def load_state(self, state: dict[str, Any]):
        """Restaura o estado do trading a partir de um dicionário de dados persistido.

        Args:
            state (Dict[str, Any]): Os dados de estado carregados.
        """
        async with self.lock:
            self.balance = state.get("balance", 0.0)
            self.is_trading = state.get("is_trading", False)

            contracts_data = state.get("active_contracts", {})
            for cid_str, c_data in contracts_data.items():
                if "status" in c_data:
                    c_data["status"] = TradeStatus[c_data["status"]]
                if "direction" in c_data:
                    c_data["direction"] = TradeDirection[c_data["direction"]]

                contract = Contract(**c_data)
                self.active_contracts[int(cid_str)] = contract
