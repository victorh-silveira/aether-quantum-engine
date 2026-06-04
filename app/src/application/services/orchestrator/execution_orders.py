"""Envio de ordens e inscricao em atualizacoes de contrato aberto."""

import logging

from .settlement_backfill import subscribe_open_contract
from .stop_win_target import resolve_stop_win_target


async def place_order(executor, symbol, direction, stake, duration=None, metrics=None):
    """Compra contrato com parametros de risco e registra assinatura de liquidacao."""
    cid = f"C{int(executor.orch._active_cycle_id):04d}"
    logger = logging.getLogger("AETH")
    params = executor.orch.config.get("risk_management", {}).get("params", {}).copy()
    if duration:
        params["duration"] = duration
    if params.get("contract_type") == "MULTIPLIER":
        target_total = resolve_stop_win_target(
            executor.orch.config.get("risk_management"), executor.orch.risk_manager.initial_bankroll
        )
        current_profit = executor.orch.risk_manager.total_session_profit
        remaining = target_total - current_profit
        if remaining > 0:
            lo = dict(params.get("limit_order") or {})
            max_tp = float(stake) * 50.0
            tp_val = min(float(remaining), max_tp)
            lo["take_profit"] = round(tp_val, 2)
            lo.pop("stop_loss", None)
            params["limit_order"] = lo
    contract = await executor.orch.trade_handler.buy_with_parameters(symbol, direction, stake, params=params)
    dur = duration or params.get("duration", 1)
    u = params.get("duration_unit", "m")
    meta = metrics if isinstance(metrics, dict) else {}
    dl_dir = meta.get("dl_direction")
    inv = bool(meta.get("direction_inverted"))
    inv_part = f" || ord={direction.name} dl={dl_dir} inv" if inv and dl_dir else f" || ord={direction.name}"
    logger.info(
        "[%s] EXEC || %s $%.2f%s || pay=%.2f cid=%s buy=$%.2f %s%s",
        cid,
        symbol,
        float(stake),
        inv_part,
        float(contract.payout),
        int(contract.contract_id),
        float(contract.buy_price),
        str(dur),
        str(u),
    )
    executor.orch.risk_manager.contract_to_symbol[contract.contract_id] = symbol
    req_timeout = float(
        executor.orch.config.get("orchestrator", {})
        .get("execution", {})
        .get("settlement_request_timeout_seconds", 30.0)
    )
    try:
        await subscribe_open_contract(executor.orch.ws, int(contract.contract_id), timeout=req_timeout)
    except Exception as e:
        logger.warning("[%s] SETTLE: subscribe cid=%s falhou: %s", cid, int(contract.contract_id), e)
    return contract
