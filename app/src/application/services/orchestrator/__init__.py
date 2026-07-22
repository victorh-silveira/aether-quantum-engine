"""Orquestrador principal: conexao Deriv, streams de velas e ciclos de trade."""

import asyncio
import logging
import time
from typing import Any

from src.application.services.auth_manager import AuthManager
from src.application.services.deep_learning.decision_bridge import (
    collect_deep_learning_decisions as collect_deep_learning_decisions,
)
from src.application.services.deep_learning.dl_retrain import tick_bars_since_train
from src.application.services.direction_loss_tracker import DirectionLossTracker, get_direction_loss_tracker
from src.application.services.orchestrator.config_symbols import normalize_symbols_and_anchor
from src.application.services.orchestrator.engine_mode import training_enabled
from src.application.services.orchestrator.execution_manager import ExecutionManager
from src.application.services.orchestrator.graceful_shutdown import close_infrastructure_connections
from src.application.services.orchestrator.orchestrator_data_signature import get_data_state_signature
from src.application.services.orchestrator.orchestrator_persistence import persist_full_state_unlocked, save_full_state
from src.application.services.orchestrator.orchestrator_run_loop import (
    run_orchestrator_main_loop,
    setup_session,
    start_streams,
    subscribe_transactions,
)
from src.application.services.orchestrator.orchestrator_settlement_queue import enqueue_contract_settlement
from src.application.services.orchestrator.orchestrator_state_restore import bar_epoch_already_processed
from src.application.services.orchestrator.post_settlement_cycle import (
    post_settlement_cycle_pending,
    run_post_settlement_breath_and_cycle,
    schedule_trading_cycle_after_settlement,
)
from src.application.services.orchestrator.settlement_backfill import reconcile_single_contract
from src.application.services.orchestrator.trading_cycle_entry import run_trading_cycle_if_ready
from src.application.services.orchestrator.trading_cycle_entry_guards import trading_cycle_entry_allowed
from src.application.services.orchestrator.training_run import run_orchestrator_training
from src.application.services.strategy.decision_mode import resolve_decision_mode
from src.domain.risk.risk_manager import RiskManager
from src.infrastructure.api.websocket_manager import WebSocketManager
from src.infrastructure.factories.infra_factory import create_infra_services
from src.infrastructure.handlers.stream_handler import StreamHandler
from src.infrastructure.handlers.trade_handler import TradeHandler
from src.infrastructure.state.json_state_store import JsonStateStore
from src.infrastructure.state.persistence_manager import PersistenceManager
from src.infrastructure.state.state_manager import StateManager
from src.infrastructure.state.trading_state import TradingState


class Orchestrator:
    """Coordena WebSocket, estado, risco e execucao por ciclo."""

    def __init__(self, config: dict, auth: AuthManager | None = None):
        """Inicializa dependencias, estado e infraestrutura do motor."""
        self.config = config
        self.infra = create_infra_services(config)
        self.state_store = self.infra.state_store
        self.market_writer = self.infra.market_writer
        self.model_store = self.infra.model_store
        mode = str(config.get("trading", {}).get("mode", "demo"))
        self.auth = auth if isinstance(auth, AuthManager) else AuthManager(mode=mode, config=config)
        timeout = int(config["api_config"]["request_timeout_seconds"])
        self.ws = WebSocketManager("", request_timeout=timeout)
        self.anchor, self.symbols = normalize_symbols_and_anchor(config)
        self.stream = StreamHandler(
            self.ws,
            self.symbols,
            config["data_handler"],
            market_writer=self.market_writer,
        )
        self.trade_handler = TradeHandler(self.ws, config, auth=self.auth)
        self.trading_transport = "ws"
        self.deriv_account_id = ""
        self.risk_manager = RiskManager(config["risk_management"])
        gran = int(config.get("data_handler", {}).get("micro_granularity", 300))
        self.risk_manager.set_candle_interval_seconds(gran)
        self.state = TradingState()
        self.persistence = JsonStateStore() if not self.infra.enabled else PersistenceManager()
        self.state_mgr = StateManager()
        self.logger = logging.getLogger("AETH")
        self.executor = ExecutionManager(self)
        self.tick_count, self.running, self.is_trading, self.lock = 0, False, False, None
        self._settlement_queue, self._settlement_worker_task, self._trading_slot_poll_task = None, None, None
        self.shutdown_reason: str | None = None
        self._cluster_results: list = []
        self._last_epoch, self._last_processed_epoch = 0, 0
        self._last_cluster_cycle_end = 0.0
        self._session_targets_bootstrapped = False
        self._buffer_result_logs = False
        self._pending_result_logs: list[str] = []
        self._cycle_seq = 0
        self._active_cycle_id = 0
        self._contract_cycle: dict[int, int] = {}
        self._last_result_cycle_id, self._last_settlement_outcome = 0, ""
        self._session_wins, self._session_losses = 0, 0
        self._stream_ready_at: float | None = None
        self._recovery_skip_counter = 0
        self._reconciliation_pending = False
        self._post_settlement_task: asyncio.Task | None = None
        self._post_settlement_wake = asyncio.Event()
        self._post_settlement_incomplete_streak, self._post_settlement_deadlock = 0, False
        self._settlement_wait_logged = False
        self._last_loss_symbol = ""
        self._last_loss_direction = ""
        self._last_idle_watchdog_attempt = 0.0
        self.last_data_signature = ""
        self._stream_ready_mono = 0.0
        self._is_initial_boot = True
        self._ingestion_watchdog = None
        self._profit_table_audit_task: asyncio.Task | None = None

    @property
    def loss_tracker(self) -> DirectionLossTracker:
        """Rastreador singleton de perdas direcionais com expiracao temporal."""
        return get_direction_loss_tracker()

    def get_data_state_signature(self) -> str:
        """Calcula uma assinatura unica do estado dos dados de mercado."""
        return get_data_state_signature(self)

    def _dl_enabled(self) -> bool:
        """Retorna se o modo decisao Deep Learning esta ativo."""
        return resolve_decision_mode(self.config) == "deep_learning"

    def _decision_mode(self) -> str:
        """Retorna o modo de decisao configurado para o ciclo."""
        return resolve_decision_mode(self.config)

    async def run_training(self) -> bool:
        """Executa sessao dedicada de treino DL e encerra."""
        self.loop = asyncio.get_running_loop()
        return await run_orchestrator_training(self)

    async def run(self):
        """Loop principal: reconexao, persistencia e ciclos por intervalo."""
        self.loop = asyncio.get_running_loop()
        await run_orchestrator_main_loop(self)

    async def _tick_idle_cycle_watchdog(self) -> None:
        """Verifica ociosidade e dispara ciclo quando necessario."""
        if not self.stream.is_synchronized or self.state.active_contracts or self.is_trading:
            return  # pragma: no cover
        task = self._post_settlement_task
        if task is not None and not task.done():
            return
        orch_cfg = self.config.get("orchestrator") if isinstance(self.config.get("orchestrator"), dict) else {}
        interval = float(orch_cfg.get("idle_cycle_watchdog_seconds", 15.0))
        if interval <= 0:
            return
        now = time.time()
        if now - self._last_idle_watchdog_attempt < interval:
            return
        self._last_idle_watchdog_attempt = now
        await self._run_trading_cycle_if_ready()

    async def _tick_interval_cycle_if_due(self) -> None:
        """Dispara ciclo completo a cada cycle_interval_seconds."""
        cycle_iv = int(self.config.get("orchestrator", {}).get("cycle_interval_seconds") or 0)
        if cycle_iv <= 0:
            return  # pragma: no cover
        if post_settlement_cycle_pending(self):
            return
        if self.stream.is_synchronized and trading_cycle_entry_allowed(self):
            await self._run_trading_cycle_if_ready()

    def mark_cluster_cycle_complete(self) -> None:
        """Atualiza timestamp de fim do ultimo cluster."""
        self._last_cluster_cycle_end = time.time()
        self.risk_manager.tick_symbol_loss_cycle_cooldowns()

    def schedule_trading_cycle_after_settlement(self) -> None:
        """Agenda novo ciclo de decisao logo apos liquidacao do contrato."""
        schedule_trading_cycle_after_settlement(self)

    async def _run_post_settlement_breath_and_cycle(self) -> None:
        """Aplica folego pos-liquidacao antes de um novo ciclo."""
        await run_post_settlement_breath_and_cycle(self)

    async def _setup_session(self) -> bool:
        """Estabelece sessao de trading com autenticacao e WebSocket."""
        return await setup_session(self)

    async def _start_streams(self) -> bool:
        """Inicia streams OHLC e sincroniza velas historicas."""
        ok = await start_streams(self)
        if ok:
            self.stream.tick_buffer.touch_activity()
            self._stream_ready_mono = asyncio.get_running_loop().time()
        return ok

    async def _on_candle(self, candle: Any):
        """Callback de vela do ancora: atualiza estado e tenta ciclo."""
        if candle.symbol != self.anchor:
            return
        if await bar_epoch_already_processed(self, self.anchor, candle.epoch):
            return
        self.tick_count += 1
        if self._last_epoch == candle.epoch:
            return
        self._last_epoch = candle.epoch
        try:
            val_epoch = int(candle.epoch)
        except (ValueError, TypeError):  # pragma: no cover
            val_epoch = int(time.time())  # pragma: no cover
        _ = val_epoch
        if training_enabled(self):
            tick_bars_since_train(self, self.symbols)
        self.risk_manager.tick_symbol_loss_cooldowns()
        self.risk_manager.decay_proposal_skip_cycles()
        if post_settlement_cycle_pending(self):
            return
        await self._run_trading_cycle_if_ready()

    async def _run_trading_cycle_if_ready(self) -> bool:
        """Executa um ciclo completo de decisao e cluster quando permitido."""
        return await run_trading_cycle_if_ready(self)

    async def _subscribe_account_transactions(self) -> None:
        """Inscreve callback de transacoes de conta na Deriv."""
        await subscribe_transactions(self)

    async def _on_transaction(self, data: dict) -> None:
        """Reconcilia contrato quando a Deriv emite transacao de fechamento."""
        txn = data.get("transaction")
        if not isinstance(txn, dict):
            return
        raw_id = txn.get("contract_id")
        if raw_id is None:
            return
        c_id = int(raw_id)
        known = (
            c_id in self.state.active_contracts
            or c_id in self.risk_manager.active_contract_ids
            or c_id in self.risk_manager.contract_to_symbol
        )
        if not known:
            return
        await reconcile_single_contract(self, c_id)

    async def _on_contract_update(self, data):
        """Enfileira liquidacao de contrato sem bloquear o loop principal."""
        await enqueue_contract_settlement(self, data)

    async def _save_full_state(self):
        """Persiste o snapshot completo do orquestrador."""
        await save_full_state(self)

    async def _persist_full_state_unlocked(self):
        """Persiste snapshot completo assumindo lock atomico ja adquirido."""
        await persist_full_state_unlocked(self)

    async def close_infrastructure_connections(self) -> None:
        """Encerra Triton, Timescale, Redis e WebSocket antes do exit."""
        await close_infrastructure_connections(self)

    async def stop(self) -> None:
        """Para o loop e fecha infraestrutura de forma graciosa."""
        await close_infrastructure_connections(self)
