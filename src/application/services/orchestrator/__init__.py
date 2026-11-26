"""Orquestrador principal: conexao Deriv, stream M5 no R_10 e decisao continua Laya."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from src.application.services.auth_manager import AuthManager
from src.application.services.orchestrator.config_symbols import normalize_symbols_and_anchor
from src.application.services.orchestrator.execution_manager import ExecutionManager
from src.application.services.orchestrator.graceful_shutdown import close_infrastructure_connections
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
from src.application.services.orchestrator.session_pair_tracker import SessionPairTracker
from src.application.services.orchestrator.settlement_backfill import reconcile_single_contract
from src.application.services.orchestrator.trading_cycle_entry import run_trading_cycle_if_ready
from src.application.services.orchestrator.trading_cycle_entry_guards import trading_cycle_entry_allowed
from src.application.services.quant_metrics_collector import QuantMetricsCollector
from src.domain.risk.risk_manager import RiskManager
from src.infrastructure.api.websocket_manager import WebSocketManager
from src.infrastructure.factories.infra_factory import create_infra_services
from src.infrastructure.handlers.stream_handler import StreamHandler
from src.infrastructure.handlers.trade_handler import TradeHandler
from src.infrastructure.inference.laya_decision_adapter import LayaDecisionAdapter
from src.infrastructure.state.json_state_store import JsonStateStore
from src.infrastructure.state.persistence_manager import PersistenceManager
from src.infrastructure.state.state_manager import StateManager
from src.infrastructure.state.trading_state import TradingState
from src.infrastructure.telemetry.metrics_server import MetricsServer
from src.infrastructure.telemetry.otel_business_instrumentor import BusinessMetricsInstrumentor


class Orchestrator:
    """Coordena WebSocket Deriv, stream M5, motor Laya e liquidacao continua."""

    def __init__(self, config: dict[str, Any], auth: AuthManager | None = None, laya: Any | None = None):
        """Inicializa dependencias, estado, infraestrutura e motor Laya."""
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
        data_cfg = dict(config.get("data_handler") or {})
        self.stream = StreamHandler(
            self.ws,
            self.symbols,
            data_cfg,
            market_writer=self.market_writer,
        )
        self.trade_handler = TradeHandler(self.ws, config, auth=self.auth, market_writer=self.market_writer)
        laya_cfg = config.get("laya") if isinstance(config.get("laya"), dict) else {}
        self.laya = (
            laya
            if laya is not None
            else LayaDecisionAdapter(
                model_id=str(laya_cfg.get("model_id", "convaiinnovations/laya")),
                device=str(laya_cfg.get("device", "auto")),
                context_candles=int(laya_cfg.get("context_candles", 12)),
                approved_checkpoint=str(laya_cfg.get("approved_checkpoint", "")),
            )
        )
        self.trading_transport = "ws"
        self.deriv_account_id = ""
        self.risk_manager = RiskManager(config.get("risk_management"), logger=logging.getLogger("AETH"))
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
        self._session_pair_tracker = SessionPairTracker()
        self._contract_cycle: dict[int, int] = {}
        self._last_result_cycle_id, self._last_settlement_outcome = 0, ""
        self._session_wins, self._session_losses = 0, 0
        self._stream_ready_at: float | None = None
        self._reconciliation_pending = False
        self._post_settlement_task: asyncio.Task | None = None
        self._post_settlement_wake = asyncio.Event()
        self._post_settlement_incomplete_streak, self._post_settlement_deadlock = 0, False
        self._settlement_wait_logged = False
        self._last_idle_watchdog_attempt = 0.0
        self.last_data_signature = ""
        self._stream_ready_mono = 0.0
        self._is_initial_boot = True
        self._streams_ever_started = False
        self._ingestion_watchdog = None
        self._profit_table_audit_task: asyncio.Task | None = None
        self.quant_collector = QuantMetricsCollector()
        self.business_metrics = BusinessMetricsInstrumentor()
        self._last_interval_cycle_check = 0.0
        self._last_cycle_was_exec_empty = False
        self._last_cycle_cluster_executed = False
        telemetry_cfg = config.get("telemetry", {}) if isinstance(config.get("telemetry"), dict) else {}
        self.metrics_server = (
            MetricsServer(
                self.quant_collector,
                instrumentor=self.business_metrics,
                host=str(telemetry_cfg.get("host", "127.0.0.1")),
                port=int(telemetry_cfg.get("port", 9100)),
            )
            if telemetry_cfg
            else None
        )

    async def _bar_epoch_already_processed(self, epoch: int) -> bool:
        """Indica se o epoch da vela M5 ja foi processado ou aberto."""
        return await bar_epoch_already_processed(self, self.anchor, epoch)

    async def save_full_state(self) -> None:
        """Persiste estado atual de sessao e contratos."""
        await save_full_state(self)

    async def _persist_full_state_unlocked(self) -> None:
        """Persiste snapshot completo sem adquirir lock externo."""
        await persist_full_state_unlocked(self)

    def _enqueue_contract_settlement(self, contract_id: int, symbol: str, expiry_time: int) -> None:
        """Enfileira contrato aberto na fila Redis ZSET de liquidacao."""
        enqueue_contract_settlement(self, contract_id, symbol, expiry_time)

    async def _reconcile_single_contract(self, contract_id: int) -> bool:
        """Reconcilia um contrato individual via broker."""
        return await reconcile_single_contract(self, contract_id)

    def mark_cluster_cycle_complete(self) -> None:
        """Libera o slot de trade e atualiza o relogio de fim de ciclo."""
        self.is_trading = False
        self._last_cluster_cycle_end = time.time()

    def _post_settlement_cycle_pending(self) -> bool:
        """Indica se ha ciclo pos-liquidacao pendente."""
        return post_settlement_cycle_pending(self)

    async def _run_post_settlement_breath_and_cycle(self) -> None:
        """Dispara novo ciclo imediatamente apos liquidacao quando pronto."""
        await run_post_settlement_breath_and_cycle(self)

    def _schedule_trading_cycle_after_settlement(self) -> None:
        """Agenda disparo do ciclo pos-liquidacao."""
        schedule_trading_cycle_after_settlement(self)

    def _trading_cycle_entry_allowed(self) -> bool:
        """Verifica se uma nova vela M5 esta pronta para avaliacao Laya."""
        return trading_cycle_entry_allowed(self)

    async def _run_trading_cycle_if_ready(self) -> bool:
        """Executa ciclo continuo com Laya quando a vela M5 estiver fechada."""
        return await run_trading_cycle_if_ready(self)

    async def _tick_idle_cycle_watchdog(self) -> None:
        """Verifica ociosidade prolongada entre ciclos M5."""
        now = time.time()
        idle_s = float((self.config.get("orchestrator") or {}).get("idle_cycle_watchdog_seconds", 1200))
        if (
            self._last_cluster_cycle_end > 0
            and (now - self._last_cluster_cycle_end) >= idle_s
            and (now - self._last_idle_watchdog_attempt) >= 60.0
        ):
            self._last_idle_watchdog_attempt = now
            self.logger.warning(
                "WATCHDOG: ciclo ocioso ha %.2fs; verificando ciclo", now - self._last_cluster_cycle_end
            )
            await self._run_trading_cycle_if_ready()

    async def _tick_interval_cycle_if_due(self) -> None:
        """Reavalia entrada continua por cadencia de vela fechada."""
        now = time.time()
        if (now - self._last_interval_cycle_check) >= 1.0:
            self._last_interval_cycle_check = now
            await self._run_trading_cycle_if_ready()

    async def setup_session(self) -> bool:
        """Autentica Deriv e prepara sessao de trading."""
        return await setup_session(self)

    async def start_streams(self) -> bool:
        """Inicia stream de velas M5 no R_10."""
        return await start_streams(self)

    async def subscribe_transactions(self) -> None:
        """Assina transacoes da conta para reconciliacao de saldo."""
        await subscribe_transactions(self)

    async def _on_candle(self, candle: Any) -> None:
        """Callback de vela M5 recebida no stream continuo."""
        _ = candle
        if self.running and not self.is_trading:
            await self._run_trading_cycle_if_ready()

    async def _on_contract_update(self, data: dict[str, Any]) -> None:
        """Callback de atualizacao de contrato aberto via WebSocket."""
        await enqueue_contract_settlement(self, data)

    async def _on_transaction(self, data: dict[str, Any]) -> None:
        """Callback de transacao da conta para reconciliar contratos ativos."""
        tx = data.get("transaction") if isinstance(data, dict) else None
        c_id = tx.get("contract_id") if isinstance(tx, dict) else None
        if c_id is not None and (
            int(c_id) in self.state.active_contracts or int(c_id) in self.risk_manager.active_contract_ids
        ):
            await self._reconcile_single_contract(int(c_id))

    def schedule_trading_cycle_after_settlement(self) -> None:
        """Agenda disparo do ciclo pos-liquidacao."""
        schedule_trading_cycle_after_settlement(self)

    async def _save_full_state(self) -> None:
        """Alias assincrono para persistencia completa de estado."""
        await self.save_full_state()

    async def run(self) -> None:
        """Loop principal continuo do orquestrador."""
        await run_orchestrator_main_loop(self)

    async def stop(self) -> None:
        """Encerra conexoes de infraestrutura do orquestrador."""
        self.running = False
        await close_infrastructure_connections(self)
