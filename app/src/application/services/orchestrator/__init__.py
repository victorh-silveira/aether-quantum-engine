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
from src.application.services.orchestrator.settlement_backfill import reconcile_single_contract
from src.application.services.orchestrator.settlement_logic import process_contract_settlement
from src.application.services.orchestrator.trading_session import trading_session_allows_entry
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
        gran = int(config.get("data_handler", {}).get("granularity", 900))
        self.risk_manager.set_candle_interval_seconds(gran)
        self.state, self.persistence = TradingState(), PersistenceManager()
        self.logger = logging.getLogger("AETH")

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
        self._risk_session_day_key: int | None = None
        self._buffer_result_logs = False
        self._pending_result_logs: list[str] = []
        self._cycle_seq = 0
        self._active_cycle_id = 0
        self._contract_cycle: dict[int, int] = {}
        self._last_result_cycle_id = 0
        self._session_wins = 0
        self._session_losses = 0
        self._last_llm_macro_tag: str | None = None
        self._last_llm_decisions: dict[str, dict] | None = None
        self._last_llm_refresh_epoch: float | None = None
        self._stream_ready_at: float | None = None
        self._post_settlement_task: asyncio.Task | None = None

    def _llm_enabled(self) -> bool:
        """Retorna se o modo decisao LLM esta ativo."""
        return bool((self.config.get("llm") or {}).get("enabled"))

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
        """Dispara ciclo completo a cada cycle_interval_seconds (macro/StatArb/EXEC)."""
        cycle_iv = int(self.config.get("orchestrator", {}).get("cycle_interval_seconds") or 0)
        if cycle_iv <= 0:
            return
        if self.stream.is_synchronized and (time.time() - self._last_cluster_cycle_end) >= cycle_iv:
            await self._run_trading_cycle_if_ready()

    def mark_cluster_cycle_complete(self) -> None:
        """Atualiza timestamp de fim do ultimo cluster."""
        self._last_cluster_cycle_end = time.time()

    def schedule_trading_cycle_after_settlement(self) -> None:
        """Agenda novo ciclo de decisao logo apos liquidacao do contrato."""
        if not self.running:
            return
        if self.state.active_contracts:
            return
        if self.is_trading:
            return
        task = self._post_settlement_task
        if task is not None and not task.done():
            return
        self._last_cluster_cycle_end = 0.0
        self._post_settlement_task = asyncio.create_task(self._run_trading_cycle_if_ready())

    async def _setup_session(self) -> bool:
        """Conecta e autoriza sessao WebSocket."""
        try:
            await self.ws.connect()
            auth = await self.ws.send({"authorize": self.token})
            if "error" in auth:
                return False
            self.state.balance = auth["authorize"]["balance"]
            self.risk_manager.set_initial_bankroll(self.state.balance)
            self._maybe_reset_daily_risk_session(int(time.time()))
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
                    self._stream_ready_at = time.time()
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

    def _maybe_reset_daily_risk_session(self, epoch: int) -> None:
        """Reinicia stop win no inicio de cada dia UTC (vela ancora)."""
        day_key = int(epoch) // 86400
        if self._risk_session_day_key == day_key:
            return
        self._risk_session_day_key = day_key
        bal = float(self.state.balance)
        self.risk_manager.reset_daily_session(bal)
        self.logger.info("RISK: Sessao diaria | banca=$%.2f | stop win diario ativo", bal)

    async def _on_candle(self, candle: Any):
        """Callback de vela do ancora: atualiza estado e tenta ciclo."""
        if candle.symbol != self.anchor:
            return
        self.tick_count += 1
        if self._last_epoch == candle.epoch:
            return
        self._last_epoch = candle.epoch
        self._maybe_reset_daily_risk_session(int(candle.epoch))
        await self._run_trading_cycle_if_ready()

    async def _run_trading_cycle_if_ready(self) -> None:
        """Executa um ciclo completo de decisao e cluster quando permitido."""
        if self.is_trading:
            return
        if self.state.active_contracts:
            self.logger.info(
                "CICLO: aguardando liquidacao (%d contrato(s) aberto(s))", len(self.state.active_contracts)
            )
            return
        epoch = int(self._last_epoch or time.time())
        ok_session, sess_note = trading_session_allows_entry(
            epoch_utc=epoch,
            stream_ready_at=self._stream_ready_at,
            now_mono=time.time(),
            config=self.config,
        )
        if not ok_session:
            self.logger.debug("CICLO: %s", sess_note)
            return
        async with self.lock:
            self.is_trading = True
            try:
                if not self.ws.is_running or not self.stream.is_synchronized:
                    self.logger.debug("STRM: aguardando sincronia...")
                    return
                self._cycle_seq += 1
                self._active_cycle_id = self._cycle_seq
                if not self._llm_enabled():
                    self.logger.error("CICLO: llm.enabled=false; motor Medallion exige LLM ativa.")
                    return
                decisions = await collect_llm_decisions(self)
                await self.executor.execute_cluster(decisions)
            except Exception as e:
                self.logger.error(f"FALHA: Ciclo: {e}")
            finally:
                self.is_trading = False

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
