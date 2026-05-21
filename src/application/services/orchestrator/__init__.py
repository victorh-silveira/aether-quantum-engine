"""Orquestrador principal: conexao Deriv, streams de velas e ciclos de trade."""

import asyncio
import logging
import time
from typing import Any

from src.application.services.llm.llm_bridge import collect_llm_decisions
from src.application.services.llm.llm_config_merge import merge_execution_section
from src.application.services.orchestrator.config_symbols import normalize_symbols_and_anchor
from src.application.services.orchestrator.decision_mode_banner import emit_decision_engine_banner
from src.application.services.orchestrator.execution_manager import ExecutionManager
from src.application.services.orchestrator.metrics_utils import stub_metrics
from src.application.services.orchestrator.settlement_backfill import reconcile_single_contract
from src.application.services.orchestrator.settlement_logic import process_contract_settlement
from src.domain.models.trade import TradeDirection
from src.domain.risk.risk_manager import RiskManager
from src.infrastructure.api.websocket_manager import WebSocketManager
from src.infrastructure.handlers.stream_handler import StreamHandler
from src.infrastructure.handlers.trade_handler import TradeHandler
from src.infrastructure.state.persistence_manager import PersistenceManager
from src.infrastructure.state.trading_state import TradingState


class Orchestrator:
    """Coordena WebSocket, estado, risco e execucao por ciclo."""

    def __init__(self, config: dict, token: str):
        merge_execution_section(config)
        self.config, self.token = config, token
        self.ws = WebSocketManager(
            config["api_config"]["base_url"], request_timeout=config["api_config"]["request_timeout_seconds"]
        )
        self.anchor, self.symbols = normalize_symbols_and_anchor(config)

        self.stream = StreamHandler(self.ws, self.symbols, config["data_handler"])
        self.trade_handler = TradeHandler(self.ws, config)
        self.risk_manager = RiskManager(config["risk_management"])
        self.state, self.persistence = TradingState(), PersistenceManager()
        self.logger = logging.getLogger("AETH")

        st = config.get("simple_trade") or {}
        self._direction_mode = str(st.get("direction_mode", "alternate")).strip().lower()
        self._alternate_next_call = True

        self.executor = ExecutionManager(self)

        self.tick_count, self.running, self.is_trading, self.lock = (
            0,
            False,
            False,
            asyncio.Lock(),
        )
        self._cluster_results = []
        self._last_epoch = 0
        self._last_cluster_cycle_end = 0.0
        self._buffer_result_logs = False
        self._pending_result_logs: list[str] = []
        self._cycle_seq = 0
        self._active_cycle_id = 0
        self._contract_cycle: dict[int, int] = {}
        self._last_result_cycle_id = 0
        self._session_wins = 0
        self._session_losses = 0

    def _llm_enabled(self) -> bool:
        """Retorna se o modo decisao LLM esta ativo."""
        return bool((self.config.get("llm") or {}).get("enabled"))

    def _stub_metrics(self, direction: TradeDirection) -> dict:
        """Metricas sinteticas alinhadas a uma direcao."""
        return stub_metrics(direction)

    def _resolve_direction_for_cycle(self) -> TradeDirection:
        """Resolve CALL/PUT conforme modo simples ou alternancia."""
        if self._direction_mode == "call":
            return TradeDirection.CALL
        if self._direction_mode == "put":
            return TradeDirection.PUT
        direction = TradeDirection.CALL if self._alternate_next_call else TradeDirection.PUT
        self._alternate_next_call = not self._alternate_next_call
        return direction

    async def run(self):
        """Loop principal: reconexao, persistencia e ciclos por intervalo."""
        if not await self._setup_session() or not await self._start_streams():
            return
        self._last_cluster_cycle_end = time.time()
        self.running = True
        emit_decision_engine_banner(self.logger, self.config, llm_enabled=self._llm_enabled())
        reconcile_counter = 0
        while self.running:
            await asyncio.sleep(1)
            if not self.ws.is_running:
                if await self._setup_session() and await self._start_streams():
                    self.logger.debug("RECOV: Sucesso.")
                else:
                    await asyncio.sleep(5)
                continue
            reconcile_counter += 1
            await self._save_full_state()

            if reconcile_counter >= self.config["orchestrator"].get("reconcile_interval_seconds", 60):
                if self.state.active_contracts:
                    await self.executor.reconcile()
                reconcile_counter = 0

            await self._tick_interval_cycle_if_due()

    async def _tick_interval_cycle_if_due(self) -> None:
        """Dispara ciclo se passou o intervalo configurado."""
        cycle_iv = int(self.config.get("orchestrator", {}).get("cycle_interval_seconds") or 0)
        if cycle_iv > 0 and self.stream.is_synchronized and (time.time() - self._last_cluster_cycle_end) >= cycle_iv:
            await self._run_trading_cycle_if_ready()

    def mark_cluster_cycle_complete(self) -> None:
        """Atualiza timestamp de fim do ultimo cluster."""
        self._last_cluster_cycle_end = time.time()

    async def _setup_session(self) -> bool:
        """Conecta e autoriza sessao WebSocket."""
        try:
            await self.ws.connect()
            auth = await self.ws.send({"authorize": self.token})
            if "error" in auth:
                return False
            self.state.balance = auth["authorize"]["balance"]
            self.risk_manager.set_initial_bankroll(self.state.balance)
            await self._subscribe_account_transactions()
            self.logger.debug(f"AUTH: Sucesso. Saldo: {self.state.balance:.2f}")
            return True
        except Exception as e:
            detalhe = str(e).strip() or repr(e)
            self.logger.error("INIT: Erro no setup [%s]: %s", type(e).__name__, detalhe, exc_info=True)
            return False

    async def _start_streams(self) -> bool:
        """Inscreve atualizacoes de contrato e inicia stream de velas."""
        self.ws.subscribe("proposal_open_contract", self._on_contract_update)
        retries = 2
        delay = 1.0
        try:
            for attempt in range(1, retries + 1):
                self.logger.debug("STRM: sincronizando velas...")
                try:
                    await self.stream.start_candle_stream(self._on_candle)
                    return True
                except ConnectionError as e:
                    if attempt >= retries:
                        raise e
                    self.logger.debug(f"STRM: reconexao durante startup ({attempt}/{retries}): {e}")
                    await asyncio.sleep(delay)
                    if not self.ws.is_running:
                        await self.ws.connect()
        except Exception as e:
            detalhe = str(e).strip() or repr(e)
            self.logger.error("STRM: Falha [%s]: %s", type(e).__name__, detalhe, exc_info=True)
            return False

    async def _on_candle(self, candle: Any):
        """Callback de vela do ancora: atualiza estado e tenta ciclo."""
        if candle.symbol != self.anchor:
            return
        self.tick_count += 1
        if self._last_epoch == candle.epoch:
            return
        self._last_epoch = candle.epoch
        await self._run_trading_cycle_if_ready()

    async def _run_trading_cycle_if_ready(self) -> None:
        """Executa um ciclo completo de decisao e cluster quando permitido."""
        if self.risk_manager.is_on_cooldown(self.tick_count) or self.is_trading:
            return
        if self.state.active_contracts:
            self.logger.debug("CICLO: aguardando liquidacao de contratos pendentes.")
            return
        async with self.lock:
            self.is_trading = True
            try:
                if not self.ws.is_running or not self.stream.is_synchronized:
                    self.logger.debug("STRM: aguardando sincronia...")
                    return
                self._cycle_seq += 1
                self._active_cycle_id = self._cycle_seq
                if self._llm_enabled():
                    decisions = await collect_llm_decisions(self)
                else:
                    decisions = self._collect_decisions()
                await self.executor.execute_cluster(decisions)
            except Exception as e:
                self.logger.error(f"FALHA: Ciclo: {e}")
            finally:
                self.is_trading = False

    def _collect_decisions(self) -> dict[str, dict]:
        """Monta mapa simbolo para direcao e metricas no modo simples."""
        direction = self._resolve_direction_for_cycle()
        metrics = self._stub_metrics(direction)
        return {sym: {"direction": direction, "metrics": metrics} for sym in self.symbols}

    async def _subscribe_account_transactions(self) -> None:
        """Inscreve notificacoes de transacao para capturar fechamento de contratos."""
        try:
            await self.ws.send({"transaction": 1, "subscribe": 1}, timeout=10)
            self.ws.subscribe("transaction", self._on_transaction)
        except Exception as e:
            self.logger.warning("SETTLE: subscribe transaction falhou: %s", e)

    async def _on_transaction(self, data: dict) -> None:
        """Reconcilia contrato quando a Deriv emite transacao de fechamento."""
        txn = data.get("transaction")
        if not isinstance(txn, dict):
            return
        raw_id = txn.get("contract_id")
        if raw_id is None:
            return
        c_id = int(raw_id)
        if c_id not in self.state.active_contracts and c_id not in self.risk_manager.active_contract_ids:
            return
        await reconcile_single_contract(self, c_id)

    async def _on_contract_update(self, data):
        """Processa liquidacao de contrato delegando para o settlement_logic."""
        await process_contract_settlement(self, data)

    async def _save_full_state(self):
        """Persiste o snapshot completo do orquestrador (Estado + Risco + PnL)."""
        s = await self.state.get_state()
        s.update(
            {
                "total_session_profit": self.risk_manager.total_session_profit,
                "risk": self.risk_manager.get_state(),
            }
        )
        self.persistence.save(s)

    async def stop(self):
        """Para o loop e fecha o WebSocket."""
        self.running = False
        await self.ws.close()
        self.logger.debug("STOP: encerrado.")
