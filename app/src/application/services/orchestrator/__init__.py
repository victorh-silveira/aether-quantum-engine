"""Orquestrador principal: conexao Deriv, streams de velas e ciclos de trade."""

import asyncio
import logging
import time
from typing import Any

from src.application.services.auth_manager import AuthManager
from src.application.services.deep_learning.decision_bridge import collect_deep_learning_decisions
from src.application.services.deep_learning.dl_deferred_train import cancel_deferred_symbol_training
from src.application.services.deep_learning.dl_retrain import tick_bars_since_train
from src.application.services.orchestrator.config_symbols import normalize_symbols_and_anchor
from src.application.services.orchestrator.decision_mode_banner import emit_decision_engine_banner
from src.application.services.orchestrator.execution_manager import ExecutionManager
from src.application.services.orchestrator.post_settlement_cycle import (
    post_settlement_cycle_pending,
    run_post_settlement_breath_and_cycle,
    schedule_trading_cycle_after_settlement,
)
from src.application.services.orchestrator.settlement_backfill import reconcile_single_contract
from src.application.services.orchestrator.settlement_logic import process_contract_settlement
from src.application.services.orchestrator.trading_cycle_entry import (
    acquire_trading_cycle_lock,
    trading_cycle_entry_allowed,
)
from src.application.services.orchestrator.ws_bootstrap import (
    setup_trading_session,
    start_orchestrator_streams,
    subscribe_account_transactions,
)
from src.domain.risk.risk_manager import RiskManager
from src.infrastructure.api.websocket_manager import WebSocketManager
from src.infrastructure.handlers.stream_handler import StreamHandler
from src.infrastructure.handlers.trade_handler import TradeHandler
from src.infrastructure.state.persistence_manager import PersistenceManager
from src.infrastructure.state.trading_state import TradingState


class Orchestrator:
    """Coordena WebSocket, estado, risco e execucao por ciclo."""

    def __init__(self, config: dict, auth: AuthManager | None = None):
        self.config = config
        mode = str(config.get("trading", {}).get("mode", "demo"))
        self.auth = auth if isinstance(auth, AuthManager) else AuthManager(mode=mode, config=config)
        timeout = int(config["api_config"]["request_timeout_seconds"])
        self.ws = WebSocketManager("", request_timeout=timeout)
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
        self.shutdown_reason: str | None = None
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
        self._stream_ready_at: float | None = None
        self._post_settlement_task: asyncio.Task | None = None
        self._post_settlement_wake = asyncio.Event()
        self._settlement_wait_logged = False
        self._last_loss_symbol = ""
        self._last_loss_direction = ""
        self._last_idle_watchdog_attempt = 0.0

    def _dl_enabled(self) -> bool:
        """Retorna se o modo decisao Deep Learning esta ativo."""
        return bool((self.config.get("deep_learning") or {}).get("enabled"))

    async def run(self):
        """Loop principal: reconexao, persistencia e ciclos por intervalo."""
        if not await self._setup_session():
            self.logger.error("INIT: Abortando motor (falha em PAT, OTP ou WebSocket).")
            return
        if not await self._start_streams():
            self.logger.error("INIT: Abortando motor (falha ao sincronizar velas OHLC).")
            return
        self._last_cluster_cycle_end = time.time()
        self.running = True
        emit_decision_engine_banner(self.logger, self.config, dl_enabled=self._dl_enabled())
        await self._run_trading_cycle_if_ready()
        reconcile_counter = 0
        orch_cfg = self.config.get("orchestrator") if isinstance(self.config.get("orchestrator"), dict) else {}
        reconnect_delay = float(orch_cfg.get("ws_reconnect_delay_seconds", 8.0))
        while self.running:
            await asyncio.sleep(1)
            if not self.ws.is_running:
                if await self._setup_session() and await self._start_streams():
                    self.logger.info("RECOV: WebSocket restaurado.")
                    reconnect_delay = float(orch_cfg.get("ws_reconnect_delay_seconds", 8.0))
                else:
                    self.logger.warning(
                        "RECOV: broker indisponivel; nova tentativa em %.0fs.",
                        reconnect_delay,
                    )
                    await asyncio.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 1.5, 60.0)
                continue
            reconcile_counter += 1
            await self._save_full_state()

            if reconcile_counter >= self.config["orchestrator"].get("reconcile_interval_seconds", 60):
                if self.state.active_contracts:
                    await self.executor.reconcile()
                reconcile_counter = 0

            await self._tick_interval_cycle_if_due()
            await self._tick_idle_cycle_watchdog()

    async def _tick_idle_cycle_watchdog(self) -> None:
        if not self.stream.is_synchronized or self.state.active_contracts or self.is_trading:
            return
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
        """Dispara ciclo completo a cada cycle_interval_seconds (macro/StatArb/EXEC)."""
        cycle_iv = int(self.config.get("orchestrator", {}).get("cycle_interval_seconds") or 0)
        if cycle_iv <= 0:
            return
        if post_settlement_cycle_pending(self):
            return
        if self.stream.is_synchronized and (time.time() - self._last_cluster_cycle_end) >= cycle_iv:
            await self._run_trading_cycle_if_ready()

    def mark_cluster_cycle_complete(self) -> None:
        """Atualiza timestamp de fim do ultimo cluster."""
        self._last_cluster_cycle_end = time.time()

    def schedule_trading_cycle_after_settlement(self) -> None:
        """Agenda novo ciclo de decisao logo apos liquidacao do contrato."""
        schedule_trading_cycle_after_settlement(self)

    async def _run_post_settlement_breath_and_cycle(self) -> None:
        """Aplica folego pos-liquidacao antes de um novo ciclo."""
        await run_post_settlement_breath_and_cycle(self)

    async def _setup_session(self) -> bool:
        return await setup_trading_session(self)

    async def _start_streams(self) -> bool:
        """Compatibilidade de testes: delega para start_orchestrator_streams."""
        return await start_orchestrator_streams(self)

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
        tick_bars_since_train(self, self.symbols)
        self.risk_manager.tick_symbol_loss_cooldowns()
        if post_settlement_cycle_pending(self):
            return
        await self._run_trading_cycle_if_ready()

    async def _run_trading_cycle_if_ready(self) -> bool:
        """Executa um ciclo completo de decisao e cluster quando permitido."""
        if not trading_cycle_entry_allowed(self) or not await acquire_trading_cycle_lock(self):
            return False
        ran = False
        try:
            if not self.ws.is_running or not self.stream.is_synchronized:
                self.logger.debug("STRM: aguardando sincronia...")
            elif not self._dl_enabled():
                self.logger.error("CICLO: deep_learning.enabled=false; motor exige Deep Learning ativo.")
                ran = True
            else:
                self._cycle_seq += 1
                if self._cycle_seq > 1:
                    self.logger.info("")
                self._active_cycle_id = self._cycle_seq
                ran = True
                self.logger.debug(
                    "[C%04d] CICLO: coletando decisoes DL (%d simbolos)",
                    self._active_cycle_id,
                    len(self.symbols),
                )
                decisions = await collect_deep_learning_decisions(self)
                await self.executor.execute_cluster(decisions)
        except Exception as e:
            self.logger.error(f"FALHA: Ciclo: {e}")
            ran = True
        finally:
            self.is_trading = False
        return ran

    async def _subscribe_account_transactions(self) -> None:
        """Compatibilidade de testes: delega para subscribe_account_transactions."""
        await subscribe_account_transactions(self)

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
        task = self._post_settlement_task
        if task is not None and not task.done():
            task.cancel()
        cancel_deferred_symbol_training(self)
        await self.ws.close()
        self.logger.debug("STOP: encerrado.")
