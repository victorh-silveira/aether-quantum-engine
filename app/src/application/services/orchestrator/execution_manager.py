"""Execucao de ordens, conciliacao e liquidacao de contratos."""

import asyncio
import logging

from src.application.services.execution_symbols import symbols_eligible_for_execution
from src.application.services.execution_symbols_recovery import pending_recovery_active
from src.application.services.force_trade_mode import force_trade_from_orch
from src.application.services.log_dedupe import clear_log_channel, log_info_if_changed
from src.application.services.orchestrator.graceful_shutdown import graceful_shutdown
from src.application.services.recovery_hurst_store import (
    prepare_recovery_skip_counter,
    reset_recovery_skip_counter_for_orch,
)
from src.domain.models.trade import TradeDirection
from src.domain.risk.stake_sizing import resolve_stake_conviction
from src.domain.risk.stop_win_target import resolve_stop_win_target

from .execution_blockers import log_execution_blockers
from .execution_collect import collect_cluster_orders
from .execution_cuda import clear_cuda_cache
from .execution_manager_execute import execute_cluster_orders
from .execution_orders import place_order
from .execution_settlement import reconcile_contracts, run_settlement_watch, wait_for_settlement


class ExecutionManager:
    """Coordena execucao de ordens e liquidacao do ciclo."""

    def __init__(self, orchestrator):
        """Inicializa dependencias do executor."""
        self.orch = orchestrator
        self.logger = logging.getLogger("AETH")

    def _include_anchor_trades(self) -> bool:
        """Indica se o simbolo ancora pode ser operado neste ciclo."""
        return bool(self.orch.config.get("orchestrator", {}).get("execution", {}).get("include_anchor_trades", False))

    def _trade_symbols(self) -> list[str]:
        """Lista simbolos elegiveis para envio de ordens no cluster."""
        return symbols_eligible_for_execution(
            self.orch.anchor, self.orch.symbols, include_anchor=self._include_anchor_trades()
        )

    def _start_result_buffer(self) -> None:
        """Ativa buffer temporario para logs de resultado."""
        self.orch._buffer_result_logs, self.orch._pending_result_logs = True, []

    def _flush_result_buffer(self) -> None:
        """Despeja logs pendentes de resultado no logger."""
        for line in self.orch._pending_result_logs:
            self.logger.info(line)
        self.orch._pending_result_logs = []

    def _cluster_stake_block(self, orders: list[tuple[str, TradeDirection, dict]], bankroll: float) -> str | None:
        """Motivo unico quando o cluster inteiro nao pode alocar stake."""
        if not orders:
            return None
        if force_trade_from_orch(self.orch):
            return None
        symbol, direction, metrics = orders[0]
        conviction = resolve_stake_conviction(metrics, self.orch.risk_manager.kelly_config)
        dl_cfg = self.orch.config.get("deep_learning", {})
        return self.orch.risk_manager.stake_block_reason(
            bankroll,
            symbol,
            conviction=conviction,
            cycle_id=int(self.orch._active_cycle_id),
            dl_metrics=metrics,
            order_direction=direction.name,
            max_val_brier=float(dl_cfg.get("max_val_brier_execute", 0.28)),
            mandatory_trade_each_cycle=self._mandatory_trade_each_cycle(),
        )

    def _log_execution_blockers(self, decisions: dict, *, pending: float = 0.0) -> None:
        """Registra motivo quando nenhuma ordem foi montada apesar de decisoes no ciclo."""
        log_execution_blockers(self, decisions, pending=pending)

    def _mandatory_trade_each_cycle(self) -> bool:
        """Indica se cada ciclo deve executar ao menos uma ordem conforme config."""
        exec_cfg = self.orch.config.get("orchestrator", {}).get("execution", {})
        if not isinstance(exec_cfg, dict):
            return False
        return bool(exec_cfg.get("mandatory_trade_each_cycle", False))

    def _collect_orders(self, decisions: dict) -> list[tuple[str, TradeDirection, dict]]:
        """Seleciona uma ordem por ciclo; modo obrigatorio ignora gate execute=false."""
        return collect_cluster_orders(self, decisions)

    async def _execute_orders(
        self, orders: list[tuple[str, TradeDirection, dict]], inter_delay: float, bankroll_snapshot: float
    ) -> int:
        """Executa ordens usando Criterio de Kelly e retorna quantidade enviada."""
        return await execute_cluster_orders(self, orders, inter_delay, bankroll_snapshot)

    async def _run_settlement_watch(self) -> None:
        """Aguarda liquidacao em background e dispara novo ciclo ao concluir."""
        await run_settlement_watch(self)

    def _training_phase_gate(self) -> bool:
        """Suspende operacao enquanto algum modelo nao concluiu o primeiro treino."""
        cid = f"C{int(self.orch._active_cycle_id):04d}"
        train_syms = getattr(self.orch, "_dl_training_symbols", frozenset())
        was_training = bool(getattr(self.orch, "_dl_training_phase", False))
        self.orch._dl_training_phase = bool(train_syms)
        if train_syms:
            ordered = [s for s in self.orch.symbols if s in train_syms] or sorted(train_syms)
            log_info_if_changed(
                self.orch,
                self.logger,
                "training_phase",
                ",".join(ordered),
                "[%s] FASE TREINO || %s | aguardando treino inicial | operacao suspensa",
                cid,
                " ".join(ordered),
            )
            return True
        if was_training:
            clear_log_channel(self.orch, "training_phase")
            self.logger.info("[%s] FASE OPERACAO || todos os modelos treinados | operacao liberada", cid)
        return False

    async def execute_cluster(self, decisions: dict):
        """Executa cluster de decisoes; liquidacao segue em background."""
        executed_count = 0
        try:
            self.orch.risk_manager.decay_proposal_skip_cycles()
            self._start_result_buffer()

            if self._training_phase_gate():
                return None

            bankroll_snapshot = float(self.orch.state.balance)
            recovery_active = pending_recovery_active(
                self.orch.risk_manager.pending_loss,
                int(getattr(self.orch.risk_manager, "consecutive_losses_linear", 0) or 0),
            )
            await prepare_recovery_skip_counter(self.orch, recovery_active=recovery_active)

            exec_chunk = self.orch.config.get("orchestrator", {}).get("execution", {})
            inter_delay = float(exec_chunk.get("inter_symbol_delay", 0.8))

            orders = self._collect_orders(decisions)
            cid = f"C{int(self.orch._active_cycle_id):04d}"
            pending = sum(self.orch.risk_manager.pending_loss.values())
            if pending > 0.0:
                sw = resolve_stop_win_target(
                    self.orch.config.get("risk_management", {}),
                    self.orch.risk_manager.initial_bankroll,
                )
                pnl = float(self.orch.risk_manager.total_session_profit)
                self.logger.debug(
                    "[%s] RISK: RECOVERY | pend=$%.2f | pnl_sessao=$%+.2f | stop_win=$%.2f",
                    cid,
                    pending,
                    pnl,
                    sw,
                )
            if not orders:
                self.orch._last_cycle_was_exec_empty = True
                self._log_execution_blockers(decisions, pending=pending)
                clear_cuda_cache()
                self.orch.is_trading = False
            else:
                self.orch._last_cycle_was_exec_empty = False
                block = self._cluster_stake_block(orders, bankroll_snapshot)
                if block:
                    self.logger.info("[%s] EXEC_PAUSE || %s", cid, block)
                    orders = []
                    self.orch._last_cycle_was_exec_empty = True
                    if block == "stop_win":
                        self.orch.shutdown_reason = "stop_win"
                        asyncio.create_task(graceful_shutdown(self.orch, fast_path=True))
                        return 0
                executed_count = await self._execute_orders(orders, inter_delay, bankroll_snapshot)
                if executed_count > 0:
                    self.orch._last_cycle_was_exec_empty = False
                    await reset_recovery_skip_counter_for_orch(self.orch)
                    self.orch.risk_manager.begin_cluster(executed_count)
                    self._flush_result_buffer()
                    self.orch._buffer_result_logs = False
                else:
                    self.orch._last_cycle_was_exec_empty = True
                    self._flush_result_buffer()
                    self.orch._buffer_result_logs = False
        finally:
            self._flush_result_buffer()
            self.orch._buffer_result_logs = False
            self.orch.mark_cluster_cycle_complete()
        if executed_count > 0:
            asyncio.create_task(self._run_settlement_watch())
        return executed_count

    def _log_exec(
        self,
        symbol,
        direction,
        stake,
        metrics,
        *,
        order_n: int = 0,
        contract_id: int | None = None,
    ):
        """Registra linha de execucao de uma ordem."""
        dur = metrics.get("duration") or self.orch.config.get("risk_management", {}).get("params", {}).get(
            "duration", 1
        )
        u = self.orch.config.get("risk_management", {}).get("params", {}).get("duration_unit", "m")
        _ = (symbol, direction, metrics, order_n)
        cid = f"C{int(self.orch._active_cycle_id):04d}"
        c_txt = str(contract_id) if contract_id is not None else "-"
        self.logger.debug(
            "[%s] ORDEM ENVIADA: $%.2f | TEMPO: %s%s | CONTRATO: %s",
            cid,
            float(stake),
            str(dur),
            str(u),
            c_txt,
        )

    async def _place_order(self, symbol, direction, stake, duration=None, metrics=None):
        """Delega compra de contrato ao modulo de ordens."""
        return await place_order(self, symbol, direction, stake, duration=duration, metrics=metrics)

    async def wait_for_settlement(self, timeout: int = 3600):
        """Monitora contratos ativos ate liquidacao ou timeout."""
        await wait_for_settlement(self, timeout=timeout)

    async def reconcile(self):
        """Consulta estado atualizado dos contratos ativos."""
        return await reconcile_contracts(self)
