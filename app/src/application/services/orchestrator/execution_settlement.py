"""Aguarda liquidacao de contratos e reconcilia estado."""

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from src.application.services.log_dedupe import clear_log_channel, log_warning_if_changed

from . import settlement_utils
from .settlement_backfill import backfill_pending_contracts, reconcile_single_contract


if TYPE_CHECKING:
    from .execution_manager import ExecutionManager


async def _settlement_poll_delay(seconds: float) -> None:
    """Aguarda intervalo de polling entre reconciliacoes de liquidacao."""
    await asyncio.sleep(seconds)


def _settlement_grace_period(exec_mgr: "ExecutionManager", execution_cfg: dict, start_time: float) -> float:
    """Calcula periodo de carencia antes de contar polls estagnados."""
    static = settlement_utils.min_elapsed_before_stagnant_polls(
        exec_mgr.orch.config.get("risk_management", {}).get("params"),
        execution_cfg,
    )
    dynamic = settlement_utils.calculate_cluster_grace_period(
        exec_mgr.orch.state.active_contracts, execution_cfg, start_time
    )
    if dynamic <= 0:
        return static
    return max(dynamic, static)


def _settlement_timed_out(exec_mgr: "ExecutionManager", start_time: float, timeout: int) -> bool:
    """Encerra rastreamento quando o timeout de liquidacao e atingido."""
    if time.time() - start_time <= timeout:
        return False
    exec_mgr.logger.error("EXEC: Timeout fatal aguardando liquidacao.")  # pragma: no cover
    settlement_utils.clear_contract_tracking(  # pragma: no cover
        list(exec_mgr.orch.risk_manager.active_contract_ids),
        exec_mgr.orch.risk_manager,  # pragma: no cover
    )  # pragma: no cover
    return True  # pragma: no cover


def _prune_orphan_settlement_ids(exec_mgr: "ExecutionManager") -> bool:
    """Remove ids orfaos e indica se o loop de liquidacao deve encerrar."""
    active_ids = list(exec_mgr.orch.risk_manager.active_contract_ids)
    kept_ids, orphan_ids = settlement_utils.prune_orphan_contract_ids(active_ids, exec_mgr.orch.state.active_contracts)
    if orphan_ids:
        exec_mgr.orch.risk_manager.active_contract_ids = kept_ids
        settlement_utils.clear_contract_metadata(orphan_ids, exec_mgr.orch.risk_manager)
    return not exec_mgr.orch.risk_manager.active_contract_ids


async def _wait_broker_offline_settlement(exec_mgr: "ExecutionManager", poll: float) -> bool:
    """Pausa liquidacao enquanto o broker estiver offline."""
    if exec_mgr.orch.ws.is_running:
        clear_log_channel(exec_mgr.orch, "settle_offline")
        return False
    pending_key = ",".join(str(x) for x in exec_mgr.orch.risk_manager.active_contract_ids)
    log_warning_if_changed(
        exec_mgr.orch,
        exec_mgr.logger,
        "settle_offline",
        pending_key,
        "SETTLE: broker offline; aguardando reconexao (pend=%s)",
        pending_key,
    )
    await _settlement_poll_delay(max(poll, 5.0))
    return True


def _next_stagnant_poll_count(
    stagnant_polls: int,
    elapsed: float,
    grace: float,
    current_ids: list[int],
    prev_active_ids: list[int],
) -> int:
    """Atualiza contador de polls estagnados apos reconciliacao."""
    if elapsed < grace:
        return 0
    if current_ids == prev_active_ids:
        return stagnant_polls + 1
    return 0


async def _handle_stagnant_settlement(
    exec_mgr: "ExecutionManager",
    poll: float,
    max_stagnant_polls: int,
    stagnant_polls: int,
) -> str | None:
    """Tenta backfill em liquidacao estagnada; retorna continue, break ou None."""
    if max_stagnant_polls <= 0 or stagnant_polls < max_stagnant_polls:
        return None
    if not exec_mgr.orch.ws.is_running:
        await _settlement_poll_delay(max(poll, 5.0))
        return "continue"
    pending = list(exec_mgr.orch.risk_manager.active_contract_ids)
    if pending:
        recovered = await backfill_pending_contracts(exec_mgr.orch, pending)
        if recovered:
            exec_mgr.logger.info("SETTLE: Recuperados %d contratos via profit_table.", recovered)
    if exec_mgr.orch.risk_manager.active_contract_ids:
        pending_key = ",".join(str(x) for x in exec_mgr.orch.risk_manager.active_contract_ids)
        log_warning_if_changed(
            exec_mgr.orch,
            exec_mgr.logger,
            "settle_stagnant",
            pending_key,
            "EXEC: Liquidacao estagnada; aguardando profit_table (pend=%s)",
            pending_key,
        )
        await _settlement_poll_delay(max(poll, 5.0))
        return "continue"
    return "break"


async def run_settlement_watch(exec_mgr: "ExecutionManager") -> None:
    """Aguarda liquidacao em background e dispara novo ciclo ao concluir."""
    try:
        await wait_for_settlement(exec_mgr)
    except Exception as e:
        exec_mgr.logger.error("SETTLE: falha no acompanhamento: %s", e)
    finally:
        if exec_mgr.orch.running and not exec_mgr.orch.state.active_contracts:
            exec_mgr.orch.schedule_trading_cycle_after_settlement()


async def wait_for_settlement(exec_mgr: "ExecutionManager", timeout: int = 3600) -> None:
    """Monitora contratos ativos ate liquidacao ou timeout."""
    start_time = time.time()
    execution_cfg = exec_mgr.orch.config.get("orchestrator", {}).get("execution", {})
    poll = float(execution_cfg.get("settlement_poll_seconds", 5.0))
    max_stagnant_polls = int(execution_cfg.get("settlement_max_stagnant_polls", 18))
    grace = _settlement_grace_period(exec_mgr, execution_cfg, start_time)
    stagnant_polls = 0
    prev_active_ids: list[int] = []

    while exec_mgr.orch.risk_manager.active_contract_ids:
        if _settlement_timed_out(exec_mgr, start_time, timeout):
            break
        if _prune_orphan_settlement_ids(exec_mgr):
            break
        if await _wait_broker_offline_settlement(exec_mgr, poll):
            stagnant_polls = 0
            continue
        if not await exec_mgr.reconcile():
            stagnant_polls = 0
            await _settlement_poll_delay(max(poll, 3.0))
            continue
        current_ids = list(exec_mgr.orch.risk_manager.active_contract_ids)
        grace = _settlement_grace_period(exec_mgr, execution_cfg, start_time)
        stagnant_polls = _next_stagnant_poll_count(
            stagnant_polls, time.time() - start_time, grace, current_ids, prev_active_ids
        )
        prev_active_ids = current_ids
        stagnant_action = await _handle_stagnant_settlement(exec_mgr, poll, max_stagnant_polls, stagnant_polls)
        if stagnant_action == "continue":
            stagnant_polls = 0
            continue
        if stagnant_action == "break":
            break
        await exec_mgr.orch._save_full_state()
        await _settlement_poll_delay(poll)
    clear_log_channel(exec_mgr.orch, "settle_stagnant")


async def reconcile_contracts(exec_mgr: "ExecutionManager") -> bool:
    """Consulta estado atualizado dos contratos ativos."""
    logger = logging.getLogger("AETH")
    ws = exec_mgr.orch.ws
    if not ws or not ws.is_running:
        return False
    for c_id in list(exec_mgr.orch.state.active_contracts.keys()):
        try:
            await reconcile_single_contract(exec_mgr.orch, int(c_id))
            await asyncio.sleep(0.2)
        except Exception as e:
            if settlement_utils.is_transient_broker_error(e):
                settlement_utils.mark_ws_offline(ws)
                logger.warning("RECONCILE: broker indisponivel cid=%s: %s", c_id, e)
                return False
            logger.warning("RECONCILE: cid=%s falhou: %s", c_id, e)
    return True
