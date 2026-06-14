"""Ponte de decisao do Deep Learning para o Orquestrador."""

import logging

from src.application.services.deep_learning.dl_bridge_helpers import (
    apply_symbol_loss_cooldown,
    build_decision_entry,
    parse_dl_params,
    pending_loss_total,
    recovery_gating_active,
)
from src.application.services.deep_learning.dl_cycle_log import log_dl_cycle_summary
from src.application.services.deep_learning.dl_deferred_train import enqueue_deferred_symbol_training
from src.application.services.deep_learning.dl_gate_config import parse_deploy_gate_config
from src.application.services.deep_learning.dl_market_data import load_symbol_close_ohlc, load_symbol_microstructure
from src.application.services.deep_learning.dl_outcomes import tick_dl_session_pauses
from src.application.services.deep_learning.dl_params import slice_dl_ohlc_window
from src.application.services.deep_learning.dl_predict import predict_symbol_decision
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
from src.application.services.orchestrator.engine_mode import training_enabled


__all__ = [
    "collect_deep_learning_decisions",
    "resolve_dl_model_path",
]

logger = logging.getLogger("AETH")

_get_symbol_runtime = get_symbol_runtime
_candle_epoch = candle_epoch
_granularity_seconds = granularity_seconds
_run_symbol_training = run_symbol_training


def _insufficient_data_entry() -> dict:
    """Monta entrada de decisao bloqueada por falta de historico de precos."""
    entry = build_decision_entry(None, 0.0, execute=False, val_accuracy=0.0, edge=0.0, train_loss=None)
    entry["metrics"]["gate_reason"] = "data"
    return entry


def _apply_deploy_gate(entry: dict, runtime: dict, dl_config: dict) -> dict:
    """Aplica bloqueio de execucao quando o mini-deploy gate reprova o modelo."""
    gate_cfg = parse_deploy_gate_config(dl_config)
    if not runtime.get("deploy_ok", False) and gate_cfg.get("enabled", True) and entry["metrics"].get("execute"):
        entry["metrics"]["execute"] = False
        entry["metrics"]["gate_reason"] = "deploy"
    entry["metrics"]["deploy_ok"] = bool(runtime.get("deploy_ok", False))
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


async def _collect_symbol_decision(
    orch,
    symbol: str,
    *,
    dl_config: dict,
    params: dict,
    min_len: int,
    granularity: int,
    recovery_active: bool,
    train_priority: frozenset[str] = frozenset(),
) -> tuple[dict, str | None]:
    """Treina (se necessario), prediz e aplica gates para um simbolo."""
    prices, open_, high, low = load_symbol_close_ohlc(orch, symbol)
    if len(prices) < min_len:
        logger.debug("DL: Historico insuficiente para %s (%d/%d velas).", symbol, len(prices), min_len)
        return _insufficient_data_entry(), None

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
    if len(prices) < min_len:
        logger.debug("DL: Historico insuficiente apos recorte para %s (%d/%d).", symbol, len(prices), min_len)
        return _insufficient_data_entry(), None

    runtime = get_symbol_runtime(orch, symbol, dl_config, params)
    epoch = candle_epoch(orch, symbol)
    train_loss = None
    train_reason = None
    if training_enabled(orch):
        do_train, reason = should_retrain_symbol(orch, symbol, runtime, params, epoch)
        if do_train and train_priority and str(symbol) not in train_priority:
            do_train, reason = False, ""
        if do_train and reason == "bootstrap":
            first_pending = next((str(s) for s in orch.symbols if str(s) in train_priority), None)
            if first_pending is not None and str(symbol) != first_pending:
                do_train, reason = False, ""
        if do_train:
            train_reason = reason
            skip_train = reason == "new_candle" and bool(getattr(orch, "_dl_fast_cycle", False))
            if skip_train or reason == "bootstrap" and getattr(orch, "_dl_bootstrap_completed", False):
                train_reason = None
            else:
                enqueue_deferred_symbol_training(
                    orch,
                    symbol,
                    train_fn=run_symbol_training,
                    train_args=(symbol, runtime, prices, dl_config, params, epoch, orch),
                    train_kwargs={
                        "granularity": granularity,
                        "open_": open_,
                        "high": high,
                        "low": low,
                        "micro": micro,
                    },
                )
    norm_stats = runtime["norm_stats"]

    entry = predict_symbol_decision(
        orch,
        symbol,
        runtime["model"],
        prices,
        norm_stats,
        runtime,
        params,
        train_loss,
        recovery_active=recovery_active,
        granularity=granularity,
        open_=open_,
        high=high,
        low=low,
        micro=micro,
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

    for symbol in orch.symbols:
        entry, reason = await _collect_symbol_decision(
            orch,
            symbol,
            dl_config=dl_config,
            params=params,
            min_len=min_len,
            granularity=granularity,
            recovery_active=recovery_active,
            train_priority=train_priority,
        )
        decisions[symbol] = entry
        if reason:
            trained.append(symbol)
            train_reason = reason

    _log_retrain_batch(trained, train_reason, params)
    log_dl_cycle_summary(
        logger,
        decisions,
        recovery_active=recovery_active,
        pending_loss_total=pending_total,
        orch=orch,
    )
    return decisions
