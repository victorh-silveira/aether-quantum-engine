"""Reconciliacao atomica de contratos apos reconexao WebSocket."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from src.application.services.orchestrator.settlement_backfill import (
    reconcile_single_contract,
    settlement_payload_from_profit_row,
)
from src.application.services.orchestrator.settlement_logic import (
    process_contract_settlement,
    process_late_settlement_from_payload,
)
from src.application.services.orchestrator.settlement_utils import is_transient_broker_error, mark_ws_offline
from src.application.services.orchestrator.settlement_ws_queries import fetch_portfolio, fetch_profit_table


@dataclass
class ReconciliationResult:
    """Resumo da auditoria pos-reconexao."""

    settled_count: int = 0
    errors: list[str] = field(default_factory=list)
    duration_ms: float = 0.0


def _reconcile_timeout(orch: Any) -> float:
    """Timeout em segundos para consultas WS de reconciliacao."""
    exec_cfg = orch.config.get("orchestrator", {}).get("execution", {})
    if not isinstance(exec_cfg, dict):
        exec_cfg = {}
    return float(exec_cfg.get("settlement_reconcile_timeout_seconds", 45.0))


def _profit_table_limit(orch: Any) -> int:
    """Limite de linhas do profit_table no burst pos-reconexao."""
    exec_cfg = orch.config.get("orchestrator", {}).get("execution", {})
    if not isinstance(exec_cfg, dict):
        exec_cfg = {}
    return int(exec_cfg.get("settlement_reconcile_profit_table_limit", 120))


def _known_contract_ids(orch: Any) -> list[int]:
    """Uniao de IDs de contratos rastreados localmente."""
    ids: set[int] = set()
    for raw in getattr(orch.state, "active_contracts", {}) or {}:
        try:
            ids.add(int(raw))
        except (TypeError, ValueError):
            continue
    for raw in getattr(orch.risk_manager, "active_contract_ids", []) or []:
        try:
            ids.add(int(raw))
        except (TypeError, ValueError):
            continue
    for raw in getattr(orch.risk_manager, "contract_to_symbol", {}) or {}:
        try:
            ids.add(int(raw))
        except (TypeError, ValueError):
            continue
    return sorted(ids)


def _open_ids_from_portfolio(portfolio: list[dict]) -> set[int]:
    """Extrai IDs de contratos ainda abertos no portfolio Deriv."""
    return {int(row.get("contract_id")) for row in portfolio if row.get("contract_id") is not None}


async def _reconcile_known_offline(
    orch: Any,
    *,
    open_ids: set[int],
    settled_ids: set[int],
    result: ReconciliationResult,
) -> None:
    """Reconcilia contratos conhecidos que nao aparecem abertos no portfolio."""
    for c_id in _known_contract_ids(orch):
        if c_id in open_ids:
            continue
        try:
            if await reconcile_single_contract(orch, c_id):
                result.settled_count += 1
                settled_ids.add(c_id)
        except Exception as exc:
            if is_transient_broker_error(exc):
                mark_ws_offline(orch.ws)
            result.errors.append(f"cid={c_id}:{type(exc).__name__}")


async def _settle_from_profit_burst(
    orch: Any,
    *,
    open_ids: set[int],
    settled_ids: set[int],
    timeout: float,
    result: ReconciliationResult,
) -> None:
    """Liquida contratos restantes via burst do profit_table."""
    remaining = [c_id for c_id in _known_contract_ids(orch) if c_id not in open_ids and c_id not in settled_ids]
    if not remaining:
        return
    rows = await fetch_profit_table(orch.ws, limit=_profit_table_limit(orch), timeout=timeout)
    row_by_id = {int(row.get("contract_id")): row for row in rows if row.get("contract_id") is not None}
    for c_id in remaining:
        row = row_by_id.get(c_id)
        if row is None:
            continue
        payload = settlement_payload_from_profit_row(c_id, row)
        poc = payload.get("proposal_open_contract")
        if not isinstance(poc, dict):
            continue
        try:
            if await orch.state.finalize_contract(c_id):
                await process_contract_settlement(orch, payload)
            else:
                await process_late_settlement_from_payload(orch, poc)
            result.settled_count += 1
        except Exception as exc:
            result.errors.append(f"profit:{c_id}:{type(exc).__name__}")


async def reconcile_after_ws_recovery(orch: Any) -> ReconciliationResult:
    """Audita contratos offline e persiste risco antes do proximo ciclo DL."""
    if getattr(orch, "_reconciliation_active", False) is True:
        return ReconciliationResult()
    orch._reconciliation_active = True

    started = time.monotonic()
    result = ReconciliationResult()
    orch._reconciliation_pending = True
    timeout = _reconcile_timeout(orch)
    settled_ids: set[int] = set()
    try:
        portfolio = await fetch_portfolio(orch.ws, timeout=timeout)
        open_ids = _open_ids_from_portfolio(portfolio)
        orch.logger.info("RECONCILE: portfolio %d abertos", len(open_ids))
        await _reconcile_known_offline(orch, open_ids=open_ids, settled_ids=settled_ids, result=result)
        await _settle_from_profit_burst(
            orch,
            open_ids=open_ids,
            settled_ids=settled_ids,
            timeout=timeout,
            result=result,
        )
        await orch._save_full_state()
    except Exception as exc:
        if is_transient_broker_error(exc):
            mark_ws_offline(orch.ws)
        result.errors.append(type(exc).__name__)
    finally:
        orch._reconciliation_pending = False
        orch._reconciliation_active = False
        result.duration_ms = (time.monotonic() - started) * 1000.0
        orch.logger.info(
            "RECONCILE: auditoria pos-RECOV concluida settled=%d erros=%d",
            result.settled_count,
            len(result.errors),
        )
    return result
