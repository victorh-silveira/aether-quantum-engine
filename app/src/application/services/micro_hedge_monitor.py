"""Monitor de cobertura delta por hedging de ticks (High-Frequency Micro-Hedging)."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from src.application.services.orchestrator.engine_supervisor import spawn_background
from src.domain.models.trade import TradeDirection


logger = logging.getLogger("AETH")


@dataclass(slots=True)
class MicroHedgeConfig:
    """Configuracao do servico de micro-hedging de ticks."""

    enabled: bool = False
    eval_window_seconds_min: float = 120.0
    eval_window_seconds_max: float = 180.0
    target_midpoint_seconds: float = 150.0
    delta_unfavorable_ticks: float = 5.0
    hedge_duration_seconds: int = 150
    hedge_stake_ratio: float = 0.50
    tick_size: float = 0.01


def load_micro_hedge_config(config: dict[str, Any] | None) -> MicroHedgeConfig:
    """Carrega configuracao de micro-hedging a partir de config global."""
    if not isinstance(config, dict):
        return MicroHedgeConfig()
    exec_cfg = config.get("orchestrator", {}).get("execution", {})
    raw = exec_cfg.get("micro_hedging") if isinstance(exec_cfg, dict) else None
    if not isinstance(raw, dict):
        return MicroHedgeConfig()
    return MicroHedgeConfig(
        enabled=bool(raw.get("enabled", False)),
        eval_window_seconds_min=float(raw.get("eval_window_seconds_min", 120.0)),
        eval_window_seconds_max=float(raw.get("eval_window_seconds_max", 180.0)),
        target_midpoint_seconds=float(raw.get("target_midpoint_seconds", 150.0)),
        delta_unfavorable_ticks=float(raw.get("delta_unfavorable_ticks", 5.0)),
        hedge_duration_seconds=int(raw.get("hedge_duration_seconds", 150)),
        hedge_stake_ratio=float(raw.get("hedge_stake_ratio", 0.50)),
        tick_size=float(raw.get("tick_size", 0.01)),
    )


def register_contract_for_hedge(
    orch: Any,
    contract_id: int,
    *,
    symbol: str,
    direction: TradeDirection,
    entry_spot: float,
    stake: float,
    duration: int = 300,
    entry_time: float | None = None,
) -> None:
    """Registra contrato primario ativo para monitoramento de hedging aos 150s."""
    if orch is None:
        return
    watch = getattr(orch, "_active_hedge_watch", None)
    if watch is None:
        orch._active_hedge_watch = {}
        watch = orch._active_hedge_watch
    watch[int(contract_id)] = {
        "symbol": str(symbol),
        "direction": direction,
        "entry_spot": float(entry_spot),
        "stake": float(stake),
        "duration": int(duration),
        "entry_time": float(entry_time if entry_time is not None else time.time()),
        "hedged": False,
    }


def is_unfavorable_drawdown(
    direction: TradeDirection,
    delta_ticks: float,
    threshold: float,
) -> bool:
    """Verifica se o drawdown em ticks violou a margem de seguranca direcional."""
    if direction in {TradeDirection.CALL, TradeDirection.MULTUP}:
        return delta_ticks <= -abs(float(threshold))
    return delta_ticks >= abs(float(threshold))


async def execute_micro_hedge_order(
    orch: Any,
    symbol: str,
    hedge_direction: TradeDirection,
    stake: float,
    duration: int,
    parent_cid: int,
    delta_ticks: float,
) -> Any:
    """Dispara a compra do micro-hedge oposto na Deriv e anexa aos contratos ativos."""
    if orch is None or getattr(orch, "trade_handler", None) is None:
        return None
    hedge_params = {
        "duration": duration,
        "duration_unit": "s",
        "contract_type": hedge_direction.value,
    }
    try:
        contract = await orch.trade_handler.buy_with_parameters(
            symbol,
            hedge_direction,
            stake,
            params=hedge_params,
        )
        if contract is not None:
            c_id = int(contract.contract_id)
            if hasattr(orch, "state") and hasattr(orch.state, "active_contracts"):
                orch.state.active_contracts[c_id] = contract
            rm = getattr(orch, "risk_manager", None)
            if rm is not None:
                rm.active_contract_ids.append(c_id)
                rm.contract_to_symbol[c_id] = symbol
                rm.contract_stakes[c_id] = stake
            logger.info(
                "HEDGE || Disparo executado: parent=%s hedge_cid=%s dir=%s stake=%.2f dur=%ds delta=%.2f ticks",
                parent_cid,
                c_id,
                hedge_direction.name,
                stake,
                duration,
                delta_ticks,
            )
            return contract
    except Exception as exc:
        logger.warning(
            "HEDGE || Falha ao executar micro-hedge para parent_cid=%s: %s",
            parent_cid,
            exc,
        )
    return None


def evaluate_hedges_once(orch: Any, now: float | None = None) -> list[int]:
    """Varre contratos sob observacao e dispara micro-hedge se o drawdown for violado aos 150s."""
    if orch is None:
        return []
    cfg = load_micro_hedge_config(getattr(orch, "config", None))
    if not cfg.enabled:
        return []
    watch = getattr(orch, "_active_hedge_watch", None)
    if not isinstance(watch, dict) or not watch:
        return []

    curr_time = float(now if now is not None else time.time())
    stream = getattr(orch, "stream", None)
    tick_buf = getattr(stream, "tick_buffer", None) if stream is not None else None
    active_state = getattr(getattr(orch, "state", None), "active_contracts", {}) or {}
    triggered: list[int] = []

    for cid, entry in list(watch.items()):
        if cid not in active_state:
            watch.pop(cid, None)
            continue
        if bool(entry.get("hedged")):
            continue

        elapsed = curr_time - float(entry["entry_time"])
        if elapsed < cfg.eval_window_seconds_min:
            continue
        if elapsed > cfg.eval_window_seconds_max:
            entry["hedged"] = True
            continue

        sym = str(entry["symbol"])
        curr_price = tick_buf.latest_price(sym) if tick_buf is not None else None
        if curr_price is None:
            continue

        delta = float(curr_price) - float(entry["entry_spot"])
        tick_sz = max(1e-6, cfg.tick_size)
        delta_ticks = delta / tick_sz
        orig_dir = entry["direction"]

        if is_unfavorable_drawdown(orig_dir, delta_ticks, cfg.delta_unfavorable_ticks):
            entry["hedged"] = True
            hedge_dir = TradeDirection.PUT if orig_dir == TradeDirection.CALL else TradeDirection.CALL
            orig_stake = float(entry["stake"])
            hedge_stake = max(1.0, round(orig_stake * cfg.hedge_stake_ratio, 2))
            triggered.append(cid)
            coro = execute_micro_hedge_order(
                orch,
                sym,
                hedge_dir,
                hedge_stake,
                cfg.hedge_duration_seconds,
                cid,
                delta_ticks,
            )
            try:
                spawn_background(orch, coro, name=f"aether-micro-hedge-{cid}")
            except RuntimeError:
                coro.close()
    return triggered


async def start_micro_hedge_monitor_worker(orch: Any, poll_interval: float = 1.0) -> None:
    """Tarefa em background que checa hedging a cada segundo enquanto o motor estiver ativo."""
    while getattr(orch, "running", False):
        try:
            evaluate_hedges_once(orch)
        except Exception as exc:
            logger.debug("HEDGE || Erro no ciclo de avaliacao de micro-hedge: %s", exc)
        await asyncio.sleep(poll_interval)
