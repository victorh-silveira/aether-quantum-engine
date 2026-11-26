"""Estado financeiro persistente para Kelly e recuperacao por payout."""

from __future__ import annotations

from typing import Any


class RiskManager:
    """Rastreia banca, perdas pendentes e liquidacoes idempotentes da sessao."""

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
        self.pending_loss = 0.0
        self.recovery_losses = 0
        self.settled_contract_ids: set[int] = set()

    def set_initial_bankroll(self, amount: float) -> None:
        """Define a banca inicial para rastreamento da sessao."""
        self.initial_bankroll = float(amount)

    def reset_session(self, bankroll: float) -> None:
        """Reinicia lucro acumulado da sessao mantendo referencia da banca."""
        self.initial_bankroll = float(bankroll)
        self.total_session_profit = 0.0
        self.peak_session_profit = 0.0
        self.pending_loss = 0.0
        self.recovery_losses = 0

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
        if cid in self.settled_contract_ids:
            return 0.0
        self.settled_contract_ids.add(cid)
        self.pending_loss = max(0.0, self.pending_loss - pnl)
        self.recovery_losses = self.recovery_losses + 1 if pnl < 0 else self.recovery_losses
        if self.pending_loss == 0:
            self.recovery_losses = 0
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
            "pending_loss": self.pending_loss,
            "recovery_losses": self.recovery_losses,
            "settled_contract_ids": sorted(self.settled_contract_ids),
        }

    def restore_state(self, payload: dict[str, Any] | None) -> None:
        """Restaura estado financeiro minimo de sessao anterior."""
        if not isinstance(payload, dict):
            return
        self.initial_bankroll = float(payload.get("initial_bankroll", self.initial_bankroll) or 0.0)
        self.total_session_profit = float(payload.get("total_session_profit", 0.0) or 0.0)
        self.peak_session_profit = float(payload.get("peak_session_profit", self.total_session_profit) or 0.0)
        self.pending_loss = max(0.0, float(payload.get("pending_loss", 0.0)))
        self.recovery_losses = max(0, int(payload.get("recovery_losses", 0)))
        self.settled_contract_ids = {int(cid) for cid in payload.get("settled_contract_ids", [])}
