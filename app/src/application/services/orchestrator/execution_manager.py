"""Execucao de ordens, conciliacao e liquidacao de contratos."""

import asyncio
import logging

from src.domain.models.trade import TradeDirection

from .execution_settlement import reconcile_contracts, run_settlement_watch, wait_for_settlement
from .settlement_backfill import subscribe_open_contract
from .stop_win_target import resolve_stop_win_target


class ExecutionManager:
    """Coordena execucao de ordens e liquidacao do ciclo."""

    def __init__(self, orchestrator):
        """Inicializa dependencias do executor."""
        self.orch = orchestrator
        self.logger = logging.getLogger("AETH")

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
        symbol, _, metrics = orders[0]
        conviction = float(metrics.get("conviction", 0.60))
        return self.orch.risk_manager.stake_block_reason(bankroll, symbol, conviction=conviction)

    def _log_execution_blockers(self, decisions: dict, *, include_anchor: bool) -> None:
        """Registra motivo quando nenhuma ordem foi montada apesar de decisoes no ciclo."""
        cid = f"C{int(self.orch._active_cycle_id):04d}"
        reasons: list[str] = []
        bankroll_snapshot = float(self.orch.state.balance)
        for symbol in self.orch.symbols:
            if symbol == self.orch.anchor and not include_anchor:
                continue
            entry = decisions.get(symbol)
            if not entry:
                continue
            metrics = entry["metrics"]
            direction = entry["direction"]
            if direction is None:
                reasons.append(f"{symbol}:sem_direcao")
                continue
            if not metrics.get("execute", True):
                reasons.append(f"{symbol}:execute_false")
                continue
            stake = self.orch.risk_manager.calculate_stake(
                bankroll_snapshot,
                symbol,
                conviction=float(metrics.get("conviction", 0.60)),
                silent=True,
                cycle_id=int(self.orch._active_cycle_id),
            )
            block = self.orch.risk_manager.stake_block_reason(
                bankroll_snapshot, symbol, conviction=float(metrics.get("conviction", 0.60))
            )
            if stake <= 0:
                reasons.append(f"{symbol}:{block or 'stake_zero'}")
        if reasons:
            self.logger.info("[%s] EXEC_NONE || %s", cid, " | ".join(reasons))

    def _collect_orders(self, decisions: dict, *, include_anchor: bool) -> list[tuple[str, TradeDirection, dict]]:
        """Filtra decisoes executaveis e retorna ordens normalizadas."""
        orders: list[tuple[str, TradeDirection, dict]] = []
        cid = f"C{int(self.orch._active_cycle_id):04d}"
        for symbol in self.orch.symbols:
            if symbol == self.orch.anchor and not include_anchor:
                continue  # pragma: no cover
            entry = decisions.get(symbol)
            if not entry:
                continue
            metrics = entry["metrics"]
            if not metrics.get("execute", True):
                self.logger.debug("[%s] SKIP: Conviccao insuficiente para %s (Metrics Gate)", cid, symbol)
                continue

            direction = entry["direction"]
            if direction is None:
                continue
            orders.append((symbol, direction, metrics))
        return orders

    async def _execute_orders(
        self, orders: list[tuple[str, TradeDirection, dict]], inter_delay: float, bankroll_snapshot: float
    ) -> int:
        """Executa ordens usando Critério de Kelly e retorna quantidade enviada."""
        executed_count = 0
        for i, (symbol, direction, metrics) in enumerate(orders):
            order_n = i + 1
            conviction = float(metrics.get("conviction", 0.60))

            stake = self.orch.risk_manager.calculate_stake(
                bankroll_snapshot,
                symbol,
                conviction=conviction,
                silent=True,
                cycle_id=int(self.orch._active_cycle_id),
            )

            if stake <= 0:
                continue

            self.orch.risk_manager.register_entry_conviction(conviction)

            if i > 0:
                await asyncio.sleep(inter_delay)
            try:
                custom_dur = metrics.get("duration")
                res = await self._place_order(symbol, direction, stake, duration=custom_dur)
                if res:
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
            include_anchor = bool(exec_chunk.get("include_anchor_trades", True))
            inter_delay = float(exec_chunk.get("inter_symbol_delay", 0.8))

            orders = self._collect_orders(decisions, include_anchor=include_anchor)
            cid = f"C{int(self.orch._active_cycle_id):04d}"
            if not orders:
                self._log_execution_blockers(decisions, include_anchor=include_anchor)
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

    async def _place_order(self, symbol, direction, stake, duration=None):
        """Compra contrato diretamente usando parâmetros para evitar rate limit de proposta."""
        cid = f"C{int(self.orch._active_cycle_id):04d}"

        params = self.orch.config.get("risk_management", {}).get("params", {}).copy()
        if duration:
            params["duration"] = duration  # pragma: no cover

        if params.get("contract_type") == "MULTIPLIER":
            target_total = resolve_stop_win_target(
                self.orch.config.get("risk_management"), self.orch.risk_manager.initial_bankroll
            )
            current_profit = self.orch.risk_manager.total_session_profit
            remaining = target_total - current_profit

            if remaining > 0:
                max_tp = float(stake) * 50.0
                tp_val = min(float(remaining), max_tp)
                params["limit_order"] = {"take_profit": round(tp_val, 2)}
                if "stop_loss" in params["limit_order"]:
                    del params["limit_order"]["stop_loss"]  # pragma: no cover

        contract = await self.orch.trade_handler.buy_with_parameters(symbol, direction, stake, params=params)
        dur = duration or params.get("duration", 1)
        u = params.get("duration_unit", "m")

        self.logger.info(
            "[%s] EXEC || %s %s $%.2f || pay=%.2f cid=%s buy=$%.2f %s%s",
            cid,
            symbol,
            direction.name,
            float(stake),
            float(contract.payout),
            int(contract.contract_id),
            float(contract.buy_price),
            str(dur),
            str(u),
        )
        self.orch.risk_manager.contract_to_symbol[contract.contract_id] = symbol
        req_timeout = float(
            self.orch.config.get("orchestrator", {})
            .get("execution", {})
            .get("settlement_request_timeout_seconds", 30.0)
        )
        try:
            await subscribe_open_contract(self.orch.ws, int(contract.contract_id), timeout=req_timeout)
        except Exception as e:
            self.logger.warning("[%s] SETTLE: subscribe cid=%s falhou: %s", cid, int(contract.contract_id), e)
        return contract

    async def wait_for_settlement(self, timeout: int = 3600):
        """Monitora contratos ativos ate liquidacao ou timeout."""
        await wait_for_settlement(self, timeout=timeout)

    async def reconcile(self):
        """Consulta estado atualizado dos contratos ativos."""
        await reconcile_contracts(self)
