"""Execucao de ordens, conciliacao e liquidacao de contratos."""

import asyncio
import logging

from src.application.services.execution_symbols import symbols_eligible_for_execution
from src.domain.models.trade import TradeDirection
from src.domain.risk.stop_win_target import resolve_stop_win_target

from .execution_blockers import log_execution_blockers
from .execution_collect import collect_cluster_orders
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
        symbol, direction, metrics = orders[0]
        conviction = float(metrics.get("trade_score", metrics.get("conviction", 0.60)))
        dl_cfg = self.orch.config.get("deep_learning", {})
        return self.orch.risk_manager.stake_block_reason(
            bankroll,
            symbol,
            conviction=conviction,
            cycle_id=int(self.orch._active_cycle_id),
            dl_metrics=metrics,
            order_direction=direction.name,
            max_val_brier=float(dl_cfg.get("max_val_brier_execute", 0.28)),
        )

    def _log_execution_blockers(self, decisions: dict) -> None:
        """Registra motivo quando nenhuma ordem foi montada apesar de decisoes no ciclo."""
        log_execution_blockers(self, decisions)

    def _execution_flags(self) -> tuple[bool, bool]:
        """Retorna (mandatory_trade_each_cycle, invert_dl_direction)."""
        exec_cfg = self.orch.config.get("orchestrator", {}).get("execution", {})
        mandatory = bool(exec_cfg.get("mandatory_trade_each_cycle", True))
        invert = bool(exec_cfg.get("invert_dl_direction", False))
        return mandatory, invert

    def _collect_orders(self, decisions: dict) -> list[tuple[str, TradeDirection, dict]]:
        """Seleciona uma ordem por ciclo; modo obrigatorio ignora gate execute=false."""
        return collect_cluster_orders(self, decisions)

    async def _execute_orders(
        self, orders: list[tuple[str, TradeDirection, dict]], inter_delay: float, bankroll_snapshot: float
    ) -> int:
        """Executa ordens usando Critério de Kelly e retorna quantidade enviada."""
        executed_count = 0
        for i, (symbol, direction, metrics) in enumerate(orders):
            order_n = i + 1
            conviction = float(metrics.get("trade_score", metrics.get("conviction", 0.60)))

            dl_cfg = self.orch.config.get("deep_learning", {})
            pending = sum(self.orch.risk_manager.pending_loss.values())
            mandatory, _ = self._execution_flags()
            stake = self.orch.risk_manager.calculate_stake(
                bankroll_snapshot,
                symbol,
                conviction=conviction,
                silent=pending <= 0.0,
                cycle_id=int(self.orch._active_cycle_id),
                dl_metrics=metrics,
                order_direction=direction.name,
                max_val_brier=float(dl_cfg.get("max_val_brier_execute", 0.28)),
                mandatory_weak_cap=mandatory and not metrics.get("execute", True),
            )

            if stake <= 0:
                continue

            self.orch.risk_manager.register_entry_conviction(conviction)

            if i > 0:
                await asyncio.sleep(inter_delay)
            try:
                custom_dur = metrics.get("duration")
                res = await self._place_order(symbol, direction, stake, duration=custom_dur, metrics=metrics)
                if res:
                    self.orch.risk_manager.record_contract_stake(int(res.contract_id), stake)
                    self.orch.risk_manager.active_contract_ids.append(res.contract_id)
                    await self.orch.state.add_contract(res)
                    self.orch._contract_cycle[int(res.contract_id)] = int(self.orch._active_cycle_id)
                    self._log_exec(
                        symbol,
                        direction,
                        stake,
                        metrics,
                        order_n=order_n,
                        contract_id=int(res.contract_id),
                    )
                    executed_count += 1
            except Exception as e:
                err_msg = str(e).lower()
                if "closed" in err_msg or "trading is not available" in err_msg:
                    self.logger.warning(f"SKIP: Sessão fechada para {symbol}: {e}")
                else:
                    self.logger.error(f"FAIL: EXEC: Falha critica na ordem {symbol}: {e}")
        return executed_count

    async def _run_settlement_watch(self) -> None:
        """Aguarda liquidacao em background e dispara novo ciclo ao concluir."""
        await run_settlement_watch(self)

    async def execute_cluster(self, decisions: dict):
        """Executa cluster de decisoes; liquidacao segue em background."""
        executed_count = 0
        try:
            self._start_result_buffer()

            bankroll_snapshot = float(self.orch.state.balance)

            exec_chunk = self.orch.config.get("orchestrator", {}).get("execution", {})
            inter_delay = float(exec_chunk.get("inter_symbol_delay", 0.8))

            orders = self._collect_orders(decisions)
            cid = f"C{int(self.orch._active_cycle_id):04d}"
            mandatory, _ = self._execution_flags()
            pending = sum(self.orch.risk_manager.pending_loss.values())
            if pending > 0.0:
                sw = resolve_stop_win_target(
                    self.orch.config.get("risk_management", {}),
                    self.orch.risk_manager.initial_bankroll,
                )
                pnl = float(self.orch.risk_manager.total_session_profit)
                self.logger.info(
                    "[%s] RISK: RECOVERY | pend=$%.2f | pnl_sessao=$%+.2f | stop_win=$%.2f",
                    cid,
                    pending,
                    pnl,
                    sw,
                )
            if not orders and not mandatory:
                self._log_execution_blockers(decisions)
            elif not orders:
                self.logger.warning("[%s] EXEC_SKIP | sem direcao inferivel em nenhum simbolo", cid)
            else:
                block = self._cluster_stake_block(orders, bankroll_snapshot)
                if block:
                    self.logger.info("[%s] EXEC_PAUSE || %s", cid, block)
                    orders = []
            executed_count = await self._execute_orders(orders, inter_delay, bankroll_snapshot)
            if executed_count > 0:
                self.orch.risk_manager.begin_cluster(executed_count)
                self._flush_result_buffer()
                self.orch._buffer_result_logs = False
            else:
                self._flush_result_buffer()
                self.orch._buffer_result_logs = False
        finally:
            self._flush_result_buffer()
            self.orch._buffer_result_logs = False
            self.orch.mark_cluster_cycle_complete()
        if executed_count > 0:
            asyncio.create_task(self._run_settlement_watch())

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
        await reconcile_contracts(self)
