"""Ponte de decisao do Deep Learning para o Orquestrador."""

import asyncio
import logging

import numpy as np

from src.application.services.deep_learning.dl_bridge_helpers import (
    apply_symbol_loss_cooldown,
    guard_inference_price_history,
    parse_dl_params,
    pending_loss_total,
    recovery_gating_active,
)
from src.application.services.deep_learning.dl_cycle_log import log_dl_cycle_summary
from src.application.services.deep_learning.dl_deferred_train import enqueue_deferred_symbol_training
from src.application.services.deep_learning.dl_gate_config import parse_deploy_gate_config
from src.application.services.deep_learning.dl_live_bar_patch import (
    patch_forming_bar_microstructure,
    patch_forming_bar_with_live_tick,
    store_patched_ohlc_snapshot,
)
from src.application.services.deep_learning.dl_market_data import load_symbol_close_ohlc, load_symbol_microstructure
from src.application.services.deep_learning.dl_outcomes import tick_dl_session_pauses
from src.application.services.deep_learning.dl_params import slice_dl_ohlc_window
from src.application.services.deep_learning.dl_predict_async import predict_symbol_decision_async
from src.application.services.deep_learning.dl_predict_build import prepare_meta_classifier_cross_symbol_bundle
from src.application.services.deep_learning.dl_retrain import should_retrain_symbol
from src.application.services.deep_learning.dl_symbol_runtime import (
    candle_epoch,
    get_symbol_runtime,
    granularity_seconds,
    resolve_dl_model_path,
)
from src.application.services.deep_learning.dl_symbol_train import run_symbol_training
from src.application.services.deep_learning.dl_training_gate import (
    min_dl_history_len as _min_dl_history_len,
    runtime_in_training,
    training_priority_symbols,
)
from src.application.services.meta_classifier_stacking import prefetch_meta_payoff_for_decisions
from src.application.services.orchestrator.engine_mode import training_enabled


predict_symbol_decision = predict_symbol_decision_async

__all__ = [
    "collect_deep_learning_decisions",
    "resolve_dl_model_path",
]

logger = logging.getLogger("AETH")

_get_symbol_runtime = get_symbol_runtime
_candle_epoch = candle_epoch
_granularity_seconds = granularity_seconds
_run_symbol_training = run_symbol_training


def _apply_deploy_gate(entry: dict, runtime: dict, dl_config: dict) -> dict:
    """Aplica bloqueio de execucao quando o mini-deploy gate reprova o modelo."""
    gate_cfg = parse_deploy_gate_config(dl_config)
    enabled = gate_cfg.get("enabled", True)
    force_ok = bool(gate_cfg.get("force_ok", False))
    deploy_ok = bool(runtime.get("deploy_ok", False)) or (not enabled) or force_ok
    if not deploy_ok and enabled and entry["metrics"].get("execute"):
        entry["metrics"]["execute"] = False
        entry["metrics"]["gate_reason"] = "deploy"
    entry["metrics"]["deploy_ok"] = bool(deploy_ok)
    return entry


def _apply_training_gate(entry: dict, runtime: dict, params: dict) -> dict:
    """Suspende execucao e marca o simbolo como em treinamento ate o primeiro treino valido."""
    if not runtime_in_training(runtime, params):
        return entry
    entry["metrics"]["execute"] = False
    entry["metrics"]["gate_reason"] = "training"
    return entry


def _log_retrain_batch(trained: list[str], train_reason: str, params: dict) -> None:
    """Registra resumo de retreino em lote no nivel DEBUG."""
    if not trained:
        return
    logger.debug(
        "DL: retreino %d simbolo(s) | %d velas | motivo=%s | %s",
        len(trained),
        int(params["training_history_bars"]),
        train_reason,
        ",".join(trained),
    )


def _maybe_schedule_training(
    orch,
    symbol: str,
    runtime: dict,
    prices: np.ndarray,
    dl_config: dict,
    params: dict,
    epoch: int,
    current_granularity: int,
    train_priority: frozenset[str],
    open_: np.ndarray | None,
    high: np.ndarray | None,
    low: np.ndarray | None,
    micro: dict[str, np.ndarray] | None,
) -> str | None:
    """Decide se agenda treino deferred para o simbolo e retorna o motivo."""
    do_train, reason = should_retrain_symbol(orch, symbol, runtime, params, epoch)
    bootstrap_train = reason == "bootstrap"
    online = bool(params.get("online_training", False))
    may_schedule_train = training_enabled(orch) or bootstrap_train or (online and do_train)
    if not may_schedule_train:
        return None

    if do_train and train_priority and str(symbol) not in train_priority:
        do_train, reason = False, ""
    if do_train and bootstrap_train:
        first_pending = next((str(s) for s in orch.symbols if str(s) in train_priority), None)
        if first_pending is not None and str(symbol) != first_pending:
            do_train, reason = False, ""

    if not do_train:
        return None

    skip_train = reason == "new_candle" and bool(getattr(orch, "_dl_fast_cycle", False))
    if skip_train or (bootstrap_train and getattr(orch, "_dl_bootstrap_completed", False)):
        return None

    enqueue_deferred_symbol_training(
        orch,
        symbol,
        train_fn=run_symbol_training,
        train_args=(symbol, runtime, prices, dl_config, params, epoch, orch),
        train_kwargs={
            "granularity": current_granularity,
            "open_": open_,
            "high": high,
            "low": low,
            "micro": micro,
        },
    )
    return reason


async def _collect_symbol_decision(
    orch,
    symbol: str,
    *,
    dl_config: dict,
    params: dict,
    min_len: int,
    granularity: int,
    train_priority: frozenset[str] = frozenset(),
) -> tuple[dict, str | None]:
    """Treina (se necessario), prediz e aplica gates para um simbolo."""
    prices_raw, open_raw, high_raw, low_raw = load_symbol_close_ohlc(
        orch,
        symbol,
        timeframe=str(params.get("train_timeframe", "macro")),
    )
    prices_raw, open_raw, high_raw, low_raw = patch_forming_bar_with_live_tick(
        orch,
        symbol,
        prices_raw,
        open_raw,
        high_raw,
        low_raw,
    )
    store_patched_ohlc_snapshot(orch, symbol, prices_raw, open_raw, high_raw, low_raw)
    runtime = get_symbol_runtime(orch, symbol, dl_config, params)
    trained_granularity = runtime.get("trained_granularity", granularity)
    micro_full = load_symbol_microstructure(orch, symbol, len(prices_raw))
    micro_full = patch_forming_bar_microstructure(orch, symbol, micro_full)
    train_bars = int(params["training_history_bars"])
    prices, open_, high, low = slice_dl_ohlc_window(
        prices_raw,
        training_history_bars=train_bars,
        open_=open_raw,
        high=high_raw,
        low=low_raw,
    )
    micro = None
    if micro_full is not None:
        micro = {k: v[-len(prices) :] for k, v in micro_full.items()}
    infer_bars = int(params.get("inference_history_bars", train_bars))
    prices_inf, open_inf, high_inf, low_inf = slice_dl_ohlc_window(
        prices_raw,
        training_history_bars=infer_bars,
        open_=open_raw,
        high=high_raw,
        low=low_raw,
    )
    micro_inf = None
    if micro_full is not None:
        micro_inf = {k: v[-len(prices_inf) :] for k, v in micro_full.items()}
    blocked = guard_inference_price_history(
        prices_raw,
        prices_inf,
        params,
        min_len=min_len,
        symbol=symbol,
        logger=logger,
    )
    if blocked is not None:
        return blocked, None
    epoch = candle_epoch(orch, symbol)
    train_loss = None
    train_reason = _maybe_schedule_training(
        orch,
        symbol,
        runtime,
        prices,
        dl_config,
        params,
        epoch,
        granularity,
        train_priority,
        open_,
        high,
        low,
        micro,
    )
    norm_stats = runtime["norm_stats"]

    entry = await predict_symbol_decision(
        orch,
        symbol,
        runtime["model"],
        prices_inf,
        norm_stats,
        runtime,
        params,
        train_loss,
        granularity=trained_granularity,
        open_=open_inf,
        high=high_inf,
        low=low_inf,
        micro=micro_inf,
    )
    entry = _apply_deploy_gate(entry, runtime, dl_config)
    entry = _apply_training_gate(entry, runtime, params)
    return apply_symbol_loss_cooldown(orch, symbol, entry), train_reason


async def collect_deep_learning_decisions(orch) -> dict[str, dict]:
    """Coleta decisoes Deep Learning para todos os simbolos do orquestrador."""
    decisions = {}
    dl_config = orch.config.get("deep_learning", {})
    if not dl_config.get("enabled", True):
        logger.warning("DL: Deep learning esta desativado na configuracao.")
        return decisions

    data_config = orch.config.get("data_handler", {})
    risk_params = orch.config.get("risk_management", {}).get("params", {})
    params = parse_dl_params(dl_config, data_config, risk_params)
    min_len = _min_dl_history_len(params)
    granularity = granularity_seconds(orch)
    tick_dl_session_pauses(orch)
    recovery_active = recovery_gating_active(orch)
    pending_total = pending_loss_total(orch)
    trained: list[str] = []
    train_reason = ""
    train_priority = training_priority_symbols(orch, dl_config, params)
    orch._dl_training_symbols = train_priority

    tasks = [
        _collect_symbol_decision(
            orch,
            symbol,
            dl_config=dl_config,
            params=params,
            min_len=min_len,
            granularity=granularity,
            train_priority=train_priority,
        )
        for symbol in orch.symbols
    ]
    results = await asyncio.gather(*tasks)
    for symbol, (entry, reason) in zip(orch.symbols, results, strict=True):
        decisions[symbol] = entry
        if reason:
            trained.append(symbol)
            train_reason = reason

    _log_retrain_batch(trained, train_reason, params)
    prepare_meta_classifier_cross_symbol_bundle(orch, decisions, params)
    await prefetch_meta_payoff_for_decisions(decisions, orch.config)
    log_dl_cycle_summary(
        logger,
        decisions,
        recovery_active=recovery_active,
        pending_loss_total=pending_total,
        orch=orch,
    )
    return decisions
