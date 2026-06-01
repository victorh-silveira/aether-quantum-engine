"""Aguarda liquidacao de contratos e reconcilia estado."""

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from . import settlement_utils
from .settlement_backfill import backfill_pending_contracts, reconcile_single_contract


if TYPE_CHECKING:
    from .execution_manager import ExecutionManager


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
    poll = float(exec_mgr.orch.config.get("orchestrator", {}).get("execution", {}).get("settlement_poll_seconds", 5.0))
    execution_cfg = exec_mgr.orch.config.get("orchestrator", {}).get("execution", {})
    max_stagnant_polls = int(execution_cfg.get("settlement_max_stagnant_polls", 18))
    stagnant_polls = 0
    prev_active_ids: list[int] = []

    grace = settlement_utils.calculate_cluster_grace_period(
        exec_mgr.orch.state.active_contracts, execution_cfg, start_time
    )

    if grace <= 0:
        grace = settlement_utils.min_elapsed_before_stagnant_polls(
            exec_mgr.orch.config.get("risk_management", {}).get("params"),
            execution_cfg,
        )

    while exec_mgr.orch.risk_manager.active_contract_ids:
        if time.time() - start_time > timeout:
            exec_mgr.logger.error("EXEC: Timeout fatal aguardando liquidacao.")  # pragma: no cover
            settlement_utils.clear_contract_tracking(
                list(exec_mgr.orch.risk_manager.active_contract_ids), exec_mgr.orch.risk_manager
            )  # pragma: no cover
            break  # pragma: no cover
        active_ids = list(exec_mgr.orch.risk_manager.active_contract_ids)
        kept_ids, orphan_ids = settlement_utils.prune_orphan_contract_ids(
            active_ids, exec_mgr.orch.state.active_contracts
        )
        if orphan_ids:
            exec_mgr.orch.risk_manager.active_contract_ids = kept_ids
            settlement_utils.clear_contract_metadata(orphan_ids, exec_mgr.orch.risk_manager)
        if not exec_mgr.orch.risk_manager.active_contract_ids:
            break
        await exec_mgr.reconcile()
        current_ids = list(exec_mgr.orch.risk_manager.active_contract_ids)
        elapsed = time.time() - start_time
        stagnant_polls = 0 if elapsed < grace else (stagnant_polls + 1 if current_ids == prev_active_ids else 0)
        prev_active_ids = current_ids
        if max_stagnant_polls > 0 and stagnant_polls >= max_stagnant_polls:
            pending = list(exec_mgr.orch.risk_manager.active_contract_ids)
            if pending:
                recovered = await backfill_pending_contracts(exec_mgr.orch, pending)
                if recovered:
                    exec_mgr.logger.info("SETTLE: Recuperados %d contratos via profit_table.", recovered)
            if exec_mgr.orch.risk_manager.active_contract_ids:
                exec_mgr.logger.warning(
                    "EXEC: Liquidacao estagnada; pendencias sem STATUS: %s",
                    ",".join(str(x) for x in exec_mgr.orch.risk_manager.active_contract_ids),
                )
                settlement_utils.clear_contract_tracking(
                    list(exec_mgr.orch.risk_manager.active_contract_ids), exec_mgr.orch.risk_manager
                )
            break

        await exec_mgr.orch._save_full_state()
        await asyncio.sleep(poll)


async def reconcile_contracts(exec_mgr: "ExecutionManager") -> None:  # pragma: no cover
    """Consulta estado atualizado dos contratos ativos."""
    logger = logging.getLogger("AETH")
    if not exec_mgr.orch.ws or not exec_mgr.orch.ws.is_running:
        return
    for c_id in list(exec_mgr.orch.state.active_contracts.keys()):
        try:
            await reconcile_single_contract(exec_mgr.orch, int(c_id))
            await asyncio.sleep(0.2)
        except Exception as e:  # pragma: no cover
            logger.warning("RECONCILE: cid=%s falhou: %s", c_id, e)  # pragma: no cover
