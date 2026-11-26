"""Execucao de ordens Laya continuas, registro em fila Redis ZSET e liquidacao."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.application.services.orchestrator.execution_contract_registration import register_confirmed_contract
from src.application.services.orchestrator.execution_orders import place_order
from src.application.services.orchestrator.execution_settlement import (
    reconcile_contracts,
    run_settlement_watch,
    wait_for_settlement,
)
from src.application.services.orchestrator.graceful_shutdown import graceful_shutdown
from src.domain.models.laya_decision import LayaDecision
from src.domain.models.trade import Contract, TradeDirection
from src.presentation.terminal.log_context import get_log_context, restore_log_context


class ExecutionManager:
    """Coordena envio continuo de ordens Rise/Fall 5m e acompanhamento de liquidacao."""

    def __init__(self, orchestrator: Any) -> None:
        self.orch = orchestrator
        self.logger = logging.getLogger("AETH")

    def _start_result_buffer(self) -> None:
        """Ativa buffer temporario para logs de liquidacao durante execucao."""
        self.orch._buffer_result_logs = True
        self.orch._pending_result_logs = []

    def _flush_result_buffer(self) -> None:
        """Despeja logs de resultado acumulados preservando contexto."""
        prev_cid, prev_sym = get_log_context()
        try:
            for line in list(getattr(self.orch, "_pending_result_logs", [])):
                self.logger.info(line)
        finally:
            restore_log_context(prev_cid, prev_sym)
        self.orch._pending_result_logs = []
        self.orch._buffer_result_logs = False

    async def execute_laya_order(self, decision: LayaDecision, stake: float, candle_epoch: int) -> bool:
        """Envia contrato Rise/Fall 5m decidido pelo Laya e inicia watch de liquidacao."""
        self._start_result_buffer()
        try:
            contract = await self._place_order(
                decision.symbol,
                decision.direction,
                float(stake),
                duration=5,
                metrics={
                    "laya_probability": decision.probability,
                    "laya_prob_call": decision.prob_call,
                    "laya_prob_put": decision.prob_put,
                    "laya_confidence": decision.confidence,
                    "laya_latency_ms": decision.latency_ms,
                    "_order_candle_epoch": candle_epoch,
                },
            )
            if contract is None:
                return False
            executed_stake = float(getattr(contract, "buy_price", 0.0) or stake)
            registered = await register_confirmed_contract(self.orch, contract, decision.symbol, executed_stake)
            if registered:
                self.orch.risk_manager.register_open_contract(
                    int(contract.contract_id), decision.symbol, executed_stake
                )
                asyncio.create_task(self._run_settlement_watch())
            return True
        finally:
            self._flush_result_buffer()

    async def _place_order(
        self,
        symbol: str,
        direction: TradeDirection,
        stake: float,
        duration: int = 5,
        metrics: dict[str, Any] | None = None,
    ) -> Contract | None:
        """Delega envio de proposta e compra ao modulo de ordens."""
        return await place_order(self, symbol, direction, stake, duration=duration, metrics=metrics)

    async def _run_settlement_watch(self) -> None:
        """Aguarda liquidacao em background e dispara novo ciclo ao concluir."""
        await run_settlement_watch(self)

    async def wait_for_settlement(self, timeout: float = 90.0) -> bool:
        """Aguarda liquidacao dos contratos ativos ate timeout."""
        return await wait_for_settlement(self, timeout=timeout)

    async def reconcile_contracts(self) -> None:
        """Reconcilia contratos pendentes com o broker."""
        await reconcile_contracts(self)

    async def shutdown(self, *, fast_path: bool = False) -> None:
        """Encerra executor via shutdown gracioso."""
        await graceful_shutdown(self, fast_path=fast_path)
