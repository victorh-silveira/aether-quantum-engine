"""Gerenciador de risco e sessao para execucao continua com Laya (1% a 5%)."""

from __future__ import annotations

from typing import Any


class RiskManager:
    """Rastreia banca da sessao, lucro realizado e contratos abertos sem vetos ou Kelly."""

    def __init__(self, config: dict[str, Any] | None = None, *, logger: Any = None) -> None:
        self.config = config or {}
        self.risk_params: dict[str, Any] = dict(self.config.get("params") or {"stake_min": 1.0})
        self.initial_bankroll: float = 0.0
        self.total_session_profit: float = 0.0
        self.peak_session_profit: float = 0.0
        self.active_contract_ids: list[int] = []
        self.contract_to_symbol: dict[int, str] = {}
        self.contract_stakes: dict[int, float] = {}
        self.contract_requested_stakes: dict[int, float] = {}
        self.cluster_results: dict[int, float] = {}
        self.expected_cluster_settlements: int = 0
        self.logger = logger

    def set_initial_bankroll(self, amount: float) -> None:
        """Define a banca inicial para rastreamento da sessao."""
        self.initial_bankroll = float(amount)

    def reset_session(self, bankroll: float) -> None:
        """Reinicia lucro acumulado da sessao mantendo referencia da banca."""
        self.initial_bankroll = float(bankroll)
        self.total_session_profit = 0.0
        self.peak_session_profit = 0.0

    def register_open_contract(self, contract_id: int, symbol: str, stake: float) -> None:
        """Registra contrato aberto e sua stake efetiva."""
        cid = int(contract_id)
        if cid not in self.active_contract_ids:
            self.active_contract_ids.append(cid)
        self.contract_to_symbol[cid] = str(symbol)
        self.contract_stakes[cid] = float(stake)

    def on_trade_settled(
        self,
        contract_id: int,
        profit: float,
        *,
        symbol: str | None = None,
    ) -> float:
        """Atualiza lucro de sessao e limpa contrato liquidado."""
        cid = int(contract_id)
        pnl = float(profit)
        if cid in self.active_contract_ids:
            self.active_contract_ids.remove(cid)
        self.contract_to_symbol.pop(cid, None)
        self.contract_stakes.pop(cid, None)
        self.contract_requested_stakes.pop(cid, None)
        self.total_session_profit += pnl
        self.peak_session_profit = max(self.peak_session_profit, self.total_session_profit)
        if symbol and self.logger is not None:
            self.logger.debug(
                "RISK_SETTLED | %s cid=%s pnl=%+.2f total=%+.2f", symbol, cid, pnl, self.total_session_profit
            )
        return pnl

    def get_state(self) -> dict[str, Any]:
        """Serializa estado financeiro minimo para persistencia de sessao."""
        return {
            "initial_bankroll": self.initial_bankroll,
            "total_session_profit": self.total_session_profit,
            "peak_session_profit": self.peak_session_profit,
        }

    def restore_state(self, payload: dict[str, Any] | None) -> None:
        """Restaura estado financeiro minimo de sessao anterior."""
        if not isinstance(payload, dict):
            return
        self.initial_bankroll = float(payload.get("initial_bankroll", self.initial_bankroll) or 0.0)
        self.total_session_profit = float(payload.get("total_session_profit", 0.0) or 0.0)
        self.peak_session_profit = float(payload.get("peak_session_profit", self.total_session_profit) or 0.0)
