"""Treino walk-forward por simbolo."""

import logging
import time

import torch

from src.application.services.deep_learning.dl_device import device_label, resolve_torch_device
from src.application.services.deep_learning.dl_features import extract_sequences
from src.application.services.deep_learning.dl_gate_config import parse_deploy_gate_config
from src.application.services.deep_learning.dl_outcomes import sample_weights_for_symbol
from src.application.services.deep_learning.dl_symbol_runtime import guard_symbol_model
from src.application.services.deep_learning.dl_symbol_train_success import apply_successful_symbol_train
from src.application.services.deep_learning.dl_training import train_model_walkforward


logger = logging.getLogger("AETH")


def _epoch_progress_logger(symbol: str, level: int, log_every: int):
    """Fabrica callback de log por epoca do treino walk-forward."""

    def _progress(epoch: int, total: int, loss_value: float, val_acc: float) -> None:
        """Registra epoca do treino quando atinge o intervalo configurado."""
        if epoch not in (1, total) and epoch % log_every != 0:
            return
        logger.log(
            level,
            "DL TREINO | %s | epoca %d/%d | loss=%.4f | val_acc=%.2f",
            symbol,
            epoch,
            total,
            loss_value,
            val_acc,
        )

    return _progress


def _log_train_insufficient(symbol: str, runtime: dict, level: int, bar_count: int) -> None:
    """Marca simbolo sem deploy e registra falta de dados para treino."""
    runtime["val_accuracy"] = 0.0
    runtime["val_brier"] = 1.0
    runtime["deploy_ok"] = False
    logger.log(
        level,
        "DL TREINO | %s | dados insuficientes (%d velas) | aguardando proximo ciclo",
        symbol,
        bar_count,
    )
    logger.log(level, "")


def _clear_cuda_after_error(exc: Exception) -> None:
    """Tenta sincronizar e liberar cache CUDA apos falha de treino na GPU."""
    if not torch.cuda.is_available() or "cuda" not in str(exc).lower():
        return
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    except Exception as cuda_cleanup_err:
        logger.debug("DL: falha ao limpar CUDA apos erro: %s", cuda_cleanup_err)


def run_symbol_training(
    symbol: str,
    runtime: dict,
    prices,
    dl_config: dict,
    params: dict,
    candle_epoch_value: int,
    orch,
    *,
    granularity: int,
    open_=None,
    high=None,
    low=None,
    micro=None,
) -> tuple[object, float | None]:
    """Executa treino walk-forward, deploy gate e persiste checkpoint."""
    model = runtime["model"]
    norm_stats = runtime["norm_stats"]
    train_loss = None
    gate_cfg = parse_deploy_gate_config(dl_config)
    level = logging.INFO
    started = time.monotonic()
    logger.log(level, "")
    train_device = resolve_torch_device(dl_config, kind="training")
    batch_size = int(params.get("training_batch_size", 512))
    log_every = max(1, int(params.get("training_log_every_n_epochs", 16)))
    logger.log(
        level,
        "DL TREINO | %s | iniciado | %d velas | ate %d epocas | device=%s | batch=%d",
        symbol,
        len(prices),
        int(params["epochs"]),
        device_label(train_device),
        batch_size,
    )
    progress_cb = _epoch_progress_logger(symbol, level, log_every)
    try:
        with guard_symbol_model(runtime):
            _, y_preview, _ = extract_sequences(
                prices,
                params["lookback"],
                granularity=granularity,
                label_horizon_bars=int(params.get("label_horizon_bars", 1)),
                label_smooth_bars=int(params.get("label_smooth_bars", 1)),
                label_mode=str(params.get("label_mode", "ma_trend")),
                label_ma_window=int(params.get("label_ma_window", 5)),
                implied_vol_bars=int(params.get("implied_vol_bars", 60)),
                symbol=str(symbol),
                open_=open_,
                high=high,
                low=low,
                micro=micro,
            )
            if len(y_preview):
                up_pct = float(y_preview.mean()) * 100.0
                logger.log(
                    level,
                    "DL TREINO | %s | amostras=%d | labels up=%.1f%% | label=%s",
                    symbol,
                    len(y_preview),
                    up_pct,
                    params.get("label_mode", "ma_trend"),
                )
            weights = sample_weights_for_symbol(
                orch,
                symbol,
                len(y_preview),
                targets=list(y_preview) if len(y_preview) else None,
            )
            train_result = train_model_walkforward(
                model,
                prices,
                params["lookback"],
                params["epochs"],
                params["lr"],
                params["validation_bars"],
                sample_weights=weights,
                weight_decay=params["weight_decay"],
                calib_ratio=params["calib_ratio"],
                granularity=granularity,
                label_horizon_bars=int(params.get("label_horizon_bars", 1)),
                label_smooth_bars=int(params.get("label_smooth_bars", 1)),
                label_mode=str(params.get("label_mode", "ma_trend")),
                label_ma_window=int(params.get("label_ma_window", 5)),
                implied_vol_bars=int(params.get("implied_vol_bars", 60)),
                symbol=str(symbol),
                open_=open_,
                high=high,
                low=low,
                micro=micro,
                batch_size=batch_size,
                dl_config=dl_config,
                progress_cb=progress_cb,
            )
            if train_result is not None:
                norm_stats, train_loss = apply_successful_symbol_train(
                    symbol,
                    runtime,
                    train_result,
                    orch=orch,
                    model=model,
                    prices=prices,
                    norm_stats=norm_stats,
                    params=params,
                    dl_config=dl_config,
                    gate_cfg=gate_cfg,
                    candle_epoch_value=candle_epoch_value,
                    granularity=granularity,
                    level=level,
                    started=started,
                    open_=open_,
                    high=high,
                    low=low,
                    micro=micro,
                )
            else:
                _log_train_insufficient(symbol, runtime, level, len(prices))
    except Exception as e:
        logger.error("DL: Erro no treinamento walk-forward para %s: %s", symbol, e)
        runtime["deploy_ok"] = False
        _clear_cuda_after_error(e)
    return norm_stats, train_loss
