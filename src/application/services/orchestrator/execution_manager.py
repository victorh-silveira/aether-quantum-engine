"""Execucao de ordens, conciliacao e liquidacao de contratos."""

import asyncio
import logging
import time

from src.domain.models.trade import TradeDirection

from . import settlement_utils
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

    def _log_cycle_idle(self) -> None:
        """Registra ciclo sem ordens executadas."""
        self.logger.debug("CICLO | id=%04d | status=IDLE | nenhuma_ordem", self.orch._active_cycle_id)

    def _emit_idle_compact_lines(self) -> None:
        """Emite BANCA em modo compacto quando nao ha ordens."""
        dur = self.orch.config.get("risk_management", {}).get("params", {}).get("duration", 1)
        u = self.orch.config.get("risk_management", {}).get("params", {}).get("duration_unit", "m")
        cid = f"C{int(self.orch._active_cycle_id):04d}"
        self.logger.debug("[%s] ORDEM ENVIADA: $0.00 | TEMPO: %s%s | CONTRATO: -", cid, str(dur), str(u))
        self.logger.info("[%s] STATUS: IDLE || P&L: $+0.00 || API: idle", cid)
        self.logger.debug(
            "[%s] BANCA FINAL: $%.2f | ACUMULADO: %dW / %dL",
            cid,
            self.orch.state.balance,
            int(self.orch._session_wins),
            int(self.orch._session_losses),
        )
        self.logger.debug("")

    def _collect_orders(self, decisions: dict, *, include_anchor: bool) -> list[tuple[str, TradeDirection, dict]]:
        """Filtra decisoes executaveis e retorna ordens normalizadas."""
        orders: list[tuple[str, TradeDirection, dict]] = []
        cid = f"C{int(self.orch._active_cycle_id):04d}"
        for symbol in self.orch.symbols:
            if symbol == self.orch.anchor and not include_anchor:
                continue
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
        cid = f"C{int(self.orch._active_cycle_id):04d}"
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
                self.logger.debug(
                    "[%s] KELLY: Edge negativo ou stake insuficiente para %s",
                    cid,
                    symbol,
                )
                continue

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
                if "trading is not available" in err_msg or "market is closed" in err_msg:
                    self.logger.warning(f"SKIP: Sessão fechada para {symbol}: {e}")
                else:
                    self.logger.error(f"FAIL: EXEC: Falha critica na ordem {symbol}: {e}")
        return executed_count

    async def execute_cluster(self, decisions: dict):
        """Executa cluster de decisoes e aguarda liquidacao quando necessario."""
        try:
            self._start_result_buffer()

            bankroll_snapshot = float(self.orch.state.balance)

            exec_chunk = self.orch.config.get("orchestrator", {}).get("execution", {})
            include_anchor = bool(exec_chunk.get("include_anchor_trades", True))
            inter_delay = float(exec_chunk.get("inter_symbol_delay", 0.8))

            orders = self._collect_orders(decisions, include_anchor=include_anchor)
            executed_count = await self._execute_orders(orders, inter_delay, bankroll_snapshot)

            if executed_count > 0:
                self.orch._buffer_result_logs = False
                self._flush_result_buffer()
                await self.wait_for_settlement()
            else:
                self._flush_result_buffer()
                self.orch._buffer_result_logs = False
                self._log_cycle_idle()
                self._emit_idle_compact_lines()
        finally:
            self._flush_result_buffer()
            self.orch._buffer_result_logs = False
            self.orch.mark_cluster_cycle_complete()

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
        """Envia proposta/compra e associa contrato ao simbolo."""
        cid = f"C{int(self.orch._active_cycle_id):04d}"

        params = self.orch.config.get("risk_management", {}).get("params", {}).copy()
        if duration:
            params["duration"] = duration

        proposal = await self._get_multiplier_proposal_with_tp(symbol, direction, stake, params)
        contract = await self.orch.trade_handler.buy_contract(proposal)
        dur = duration or params.get("duration", 1)
        u = params.get("duration_unit", "m")
        self.logger.info(
            "[%s] EXEC || %s %s $%.2f || pay=%.2f sp=%.5f cid=%s buy=$%.2f %s%s",
            cid,
            symbol,
            direction.name,
            float(stake),
            float(proposal.payout),
            float(proposal.spot),
            int(contract.contract_id),
            float(contract.buy_price),
            str(dur),
            str(u),
        )
        self.orch.risk_manager.contract_to_symbol[contract.contract_id] = symbol
        return contract

    async def _get_multiplier_proposal_with_tp(self, symbol, direction, stake, params):
        """Calcula TP dinâmico para atingir stop win de 3% da banca."""
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
                    del params["limit_order"]["stop_loss"]

        return await self.orch.trade_handler.get_proposal(symbol, direction, stake, params=params)

    async def wait_for_settlement(self, timeout: int = 3600):
        """Monitora contratos ativos ate liquidacao ou timeout."""
        start_time = time.time()
        poll = float(self.orch.config.get("orchestrator", {}).get("execution", {}).get("settlement_poll_seconds", 5.0))
        execution_cfg = self.orch.config.get("orchestrator", {}).get("execution", {})
        max_stagnant_polls = int(execution_cfg.get("settlement_max_stagnant_polls", 18))
        stagnant_polls = 0
        prev_active_ids: list[int] = []

        grace = settlement_utils.calculate_cluster_grace_period(
            self.orch.state.active_contracts, execution_cfg, start_time
        )

        if grace <= 0:
            grace = settlement_utils.min_elapsed_before_stagnant_polls(
                self.orch.config.get("risk_management", {}).get("params"),
                execution_cfg,
            )

        while self.orch.risk_manager.active_contract_ids:
            if time.time() - start_time > timeout:
                self.logger.error("EXEC: Timeout fatal aguardando liquidacao.")
                settlement_utils.clear_contract_tracking(
                    list(self.orch.risk_manager.active_contract_ids), self.orch.risk_manager
                )
                break
            active_ids = list(self.orch.risk_manager.active_contract_ids)
            kept_ids, orphan_ids = settlement_utils.prune_orphan_contract_ids(
                active_ids, self.orch.state.active_contracts
            )
            if orphan_ids:
                self.orch.risk_manager.active_contract_ids = kept_ids
                settlement_utils.clear_contract_metadata(orphan_ids, self.orch.risk_manager)
            if not self.orch.risk_manager.active_contract_ids:
                break
            await self.reconcile()
            current_ids = list(self.orch.risk_manager.active_contract_ids)
            elapsed = time.time() - start_time
            stagnant_polls = 0 if elapsed < grace else (stagnant_polls + 1 if current_ids == prev_active_ids else 0)
            prev_active_ids = current_ids
            if max_stagnant_polls > 0 and stagnant_polls >= max_stagnant_polls:
                self.logger.warning("EXEC: Liquidacao estagnada; mantendo pendencias e aguardando reconciliacao.")
                break

            await self.orch._save_full_state()
            await asyncio.sleep(poll)

    async def reconcile(self):
        """Consulta estado atualizado dos contratos ativos."""
        req_timeout = float(
            self.orch.config.get("orchestrator", {}).get("execution", {}).get("settlement_request_timeout_seconds", 8.0)
        )
        for c_id in list(self.orch.state.active_contracts.keys()):
            try:
                res = await self.orch.ws.send({"proposal_open_contract": 1, "contract_id": c_id}, timeout=req_timeout)
                if "proposal_open_contract" in res:
                    await self.orch._on_contract_update(res)
                await asyncio.sleep(0.2)
            except Exception as e:
                self.logger.debug(f"RECONCILE: Check {c_id} falhou: {e}")
