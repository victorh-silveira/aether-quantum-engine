"""Treino inicial sequencial de todos os simbolos antes da primeira operacao."""

from __future__ import annotations

import asyncio
import logging

from src.application.services.deep_learning.dl_bridge_helpers import parse_dl_params
from src.application.services.deep_learning.dl_market_data import load_symbol_close_ohlc, load_symbol_microstructure
from src.application.services.deep_learning.dl_params import slice_dl_ohlc_window
from src.application.services.deep_learning.dl_symbol_runtime import (
    candle_epoch,
    get_symbol_runtime,
    granularity_seconds,
)
from src.application.services.deep_learning.dl_symbol_train import run_symbol_training
from src.application.services.deep_learning.dl_training_gate import (
    min_dl_history_len,
    resolve_train_ready_bars,
    runtime_in_training,
    training_priority_symbols,
)
from src.application.services.orchestrator.config_symbols import resolve_dl_train_symbols


logger = logging.getLogger("AETH")

_STATUS_WAIT = "wait"
_STATUS_OK = "ok"
_STATUS_FAIL = "fail"
_HISTORY_WAIT_CAP_SECONDS = 30.0


def _history_wait_seconds(granularity: int, dl_config: dict | None = None) -> float:
    """Espera curta entre retries de historico; nao escala com M15 (900s)."""
    cfg = dl_config if isinstance(dl_config, dict) else {}
    try:
        cap = float(cfg.get("bootstrap_history_wait_cap_seconds", _HISTORY_WAIT_CAP_SECONDS))
    except (TypeError, ValueError):
        cap = _HISTORY_WAIT_CAP_SECONDS
    cap = max(1.0, min(120.0, cap))
    return min(float(max(1, int(granularity))), cap)


def _ordered_bootstrap_symbols(orch) -> list[str]:
    """Lista simbolos pendentes de primeiro treino na ordem configurada."""
    dl_config = orch.config.get("deep_learning", {})
    data_config = orch.config.get("data_handler", {})
    risk_params = orch.config.get("risk_management", {}).get("params", {})
    params = parse_dl_params(dl_config, data_config, risk_params)
    pending = training_priority_symbols(orch, dl_config, params)
    if not pending:
        return []
    trainable = resolve_dl_train_symbols(orch.config)
    return [str(symbol) for symbol in trainable if str(symbol) in pending]


def _bootstrap_training_context(orch, symbol: str):
    """Carrega config, runtime e OHLC necessarios para treinar um simbolo."""
    dl_config = orch.config.get("deep_learning", {})
    data_config = orch.config.get("data_handler", {})
    risk_params = orch.config.get("risk_management", {}).get("params", {})
    params = parse_dl_params(dl_config, data_config, risk_params)
    train_bars = int(params.get("training_history_bars", 0) or 0)
    min_len = max(min_dl_history_len(params), train_bars)
    granularity = granularity_seconds(orch)
    train_tf = str(params.get("train_timeframe", "macro"))
    runtime = get_symbol_runtime(orch, symbol, dl_config, params)
    prices, open_, high, low = load_symbol_close_ohlc(orch, symbol, timeframe=train_tf)
    micro_full = load_symbol_microstructure(orch, symbol, len(prices))
    prices, open_, high, low = slice_dl_ohlc_window(
        prices,
        training_history_bars=int(params["training_history_bars"]),
        open_=open_,
        high=high,
        low=low,
    )
    micro = None
    if micro_full is not None:
        micro = {k: v[-len(prices) :] for k, v in micro_full.items()}
    return dl_config, params, min_len, granularity, runtime, prices, open_, high, low, micro


async def _train_bootstrap_symbol(orch, symbol: str) -> str:
    """Treina um simbolo; retorna wait|ok|fail."""
    ctx = _bootstrap_training_context(orch, symbol)
    dl_config, params, min_len, granularity, runtime, prices, open_, high, low, micro = ctx
    ready, want, soft = resolve_train_ready_bars(params, len(prices))
    if not ready:
        logger.warning(
            "DL TREINO | %s | historico insuficiente (%d/%d velas) | aguardando proximo ciclo",
            symbol,
            len(prices),
            want if want > 0 else min_len,
        )
        return _STATUS_WAIT
    if soft:
        logger.info(
            "DL TREINO | %s | historico parcial API (%d/%d velas) — seguindo",
            symbol,
            len(prices),
            want,
        )
    epoch = candle_epoch(orch, symbol, timeframe=str(params.get("train_timeframe", "macro")))
    await asyncio.to_thread(
        run_symbol_training,
        symbol,
        runtime,
        prices,
        dl_config,
        params,
        epoch,
        orch,
        granularity=granularity,
        open_=open_,
        high=high,
        low=low,
        micro=micro,
    )
    if bool(runtime.get("export_ok")):
        return _STATUS_OK
    logger.error(
        "DL TREINO | %s | deploy_ok=false (export_ok=false) — nao iniciar meta",
        symbol,
    )
    return _STATUS_FAIL


async def run_initial_bootstrap_training(orch) -> None:
    """Treina todos os modelos pendentes em sequencia antes do primeiro ciclo de execucao."""
    pending = _ordered_bootstrap_symbols(orch)
    if not pending:
        return
    logger.info("")
    logger.info(
        "DL | TREINO INICIAL | %d modelo(s) | sequencial | operacao suspensa",
        len(pending),
    )
    gran = max(1, granularity_seconds(orch))
    dl_config = orch.config.get("deep_learning", {})
    max_wait_rounds = max(1, int(dl_config.get("bootstrap_max_wait_rounds", 120)))
    wait_rounds = 0
    failed: set[str] = set()
    while pending:
        progress = False
        actionable = False
        for symbol in list(pending):
            dl_config, params, _, _, runtime, _, _, _, _, _ = _bootstrap_training_context(orch, symbol)
            if not runtime_in_training(runtime, params):
                continue
            actionable = True
            status = await _train_bootstrap_symbol(orch, symbol)
            if status == _STATUS_OK:
                progress = True
            elif status == _STATUS_FAIL:
                failed.add(symbol)
        pending = [s for s in _ordered_bootstrap_symbols(orch) if s not in failed]
        if not pending or failed:
            break
        if not progress and not actionable:
            break
        if not progress:
            wait_rounds += 1
            _, params, min_len, _, _, _, _, _, _, _ = _bootstrap_training_context(orch, pending[0])
            await orch.stream.ensure_cluster_history(min_len, timeframe=str(params.get("train_timeframe", "macro")))
            if wait_rounds >= max_wait_rounds:
                logger.warning(
                    "DL TREINO | bootstrap | limite de %d ciclos aguardando historico",
                    max_wait_rounds,
                )
                break
            await asyncio.sleep(_history_wait_seconds(gran, dl_config))


async def run_dl_training_session(orch) -> bool:
    """Treina simbolos configurados; False se algum export falhar ou ficar incompleto."""
    symbols = resolve_dl_train_symbols(orch.config)
    if not symbols:
        return True
    logger.info("")
    logger.info(
        "DL | SESSAO TREINO | %d simbolo(s) | sequencial",
        len(symbols),
    )
    gran = max(1, granularity_seconds(orch))
    dl_config = orch.config.get("deep_learning", {})
    max_wait_rounds = max(1, int(dl_config.get("bootstrap_max_wait_rounds", 120)))
    wait_rounds = 0
    completed: set[str] = set()
    failed: set[str] = set()
    while len(completed) + len(failed) < len(symbols):
        progress = False
        for symbol in symbols:
            if symbol in completed or symbol in failed:
                continue
            status = await _train_bootstrap_symbol(orch, symbol)
            if status == _STATUS_OK:
                completed.add(symbol)
                progress = True
            elif status == _STATUS_FAIL:
                failed.add(symbol)
        if failed:
            break
        if len(completed) >= len(symbols):
            break
        if not progress:
            wait_rounds += 1
            _, params, min_len, _, _, _, _, _, _, _ = _bootstrap_training_context(orch, symbols[0])
            await orch.stream.ensure_cluster_history(min_len, timeframe=str(params.get("train_timeframe", "macro")))
            if wait_rounds >= max_wait_rounds:
                logger.warning(
                    "DL TREINO | sessao | limite de %d ciclos aguardando historico",
                    max_wait_rounds,
                )
                break
            await asyncio.sleep(_history_wait_seconds(gran, dl_config))
    ok = not failed and len(completed) == len(symbols)
    if not ok:
        logger.error(
            "DL | sessao INCOMPLETA | export_ok=%d falha=%d pendente=%d — meta abortado",
            len(completed),
            len(failed),
            len(symbols) - len(completed) - len(failed),
        )
    return ok
