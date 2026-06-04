"""Ponte de decisao do Deep Learning para o Orquestrador."""

import asyncio
import logging

from src.application.services.deep_learning.dl_bridge_helpers import (
    apply_symbol_loss_cooldown,
    build_decision_entry,
    parse_dl_params,
    pending_loss_total,
    recovery_gating_active,
)
from src.application.services.deep_learning.dl_cycle_log import log_dl_cycle_summary
from src.application.services.deep_learning.dl_gate_config import parse_deploy_gate_config
from src.application.services.deep_learning.dl_outcomes import tick_dl_session_pauses
from src.application.services.deep_learning.dl_params import slice_dl_price_window
from src.application.services.deep_learning.dl_predict import predict_symbol_decision
from src.application.services.deep_learning.dl_retrain import should_retrain_symbol
from src.application.services.deep_learning.dl_symbol_runtime import (
    candle_epoch,
    get_symbol_runtime,
    granularity_seconds,
    pair_prices_for_symbol,
    resolve_dl_model_path,
    run_symbol_training,
)


__all__ = [
    "collect_deep_learning_decisions",
    "resolve_dl_model_path",
]

logger = logging.getLogger("AETH")

_pair_prices_for_symbol = pair_prices_for_symbol
_get_symbol_runtime = get_symbol_runtime
_candle_epoch = candle_epoch
_granularity_seconds = granularity_seconds
_run_symbol_training = run_symbol_training


async def collect_deep_learning_decisions(orch) -> dict[str, dict]:
    """Coleta decisoes Deep Learning para todos os simbolos do orquestrador."""
    decisions = {}
    dl_config = orch.config.get("deep_learning", {})
    if not dl_config.get("enabled", True):
        logger.warning("DL: Deep learning esta desativado na configuracao.")
        return decisions

    data_config = orch.config.get("data_handler", {})
    params = parse_dl_params(dl_config, data_config)
    min_operational = params["lookback"] + params["validation_bars"] + 20
    min_len = max(min_operational, int(params["training_history_bars"]))
    granularity = granularity_seconds(orch)
    tick_dl_session_pauses(orch)
    recovery_active = recovery_gating_active(orch)
    pending_total = pending_loss_total(orch)

    for symbol in orch.symbols:
        prices = orch.stream.get_numpy_series(symbol, "close")
        if len(prices) < min_len:
            logger.info("DL: Historico insuficiente para %s (%d/%d velas).", symbol, len(prices), min_len)
            entry = build_decision_entry(None, 0.0, execute=False, val_accuracy=0.0, edge=0.0, train_loss=None)
            entry["metrics"]["gate_reason"] = "data"
            decisions[symbol] = entry
            continue

        pair_prices = pair_prices_for_symbol(orch, symbol)
        prices, pair_prices = slice_dl_price_window(
            prices,
            pair_prices,
            training_history_bars=int(params["training_history_bars"]),
        )

        runtime = get_symbol_runtime(orch, symbol, dl_config, params)
        epoch = candle_epoch(orch, symbol)
        train_loss = None
        do_train, reason = should_retrain_symbol(orch, symbol, runtime, params, epoch)
        if do_train:
            logger.info("DL: Treinando %s | %d velas | motivo=%s", symbol, len(prices), reason)
            norm_stats, train_loss = await asyncio.to_thread(
                run_symbol_training,
                symbol,
                runtime,
                prices,
                dl_config,
                params,
                epoch,
                orch,
                pair_prices=pair_prices,
                granularity=granularity,
            )
        else:
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
            pair_prices=pair_prices,
        )
        gate_cfg = parse_deploy_gate_config(dl_config)
        if not runtime.get("deploy_ok", False) and gate_cfg.get("enabled", True) and entry["metrics"].get("execute"):
            entry["metrics"]["execute"] = False
            entry["metrics"]["gate_reason"] = "deploy"
        entry["metrics"]["deploy_ok"] = bool(runtime.get("deploy_ok", False))
        decisions[symbol] = apply_symbol_loss_cooldown(orch, symbol, entry)

    log_dl_cycle_summary(
        logger,
        decisions,
        recovery_active=recovery_active,
        pending_loss_total=pending_total,
    )
    return decisions
