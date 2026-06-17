"""Runtime de modelo e checkpoints por simbolo."""

import logging
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from aether_paths import repo_path
from src.application.services.deep_learning.dl_calibration import CalibratorState
from src.application.services.deep_learning.dl_device import log_device_once, place_model, resolve_torch_device
from src.application.services.deep_learning.dl_features import FEATURE_DIM
from src.application.services.deep_learning.model import (
    create_direction_model,
    fit_norm_stats,
    load_model_checkpoint,
)


logger = logging.getLogger("AETH")


@contextmanager
def guard_symbol_model(runtime: dict):
    """Serializa treino e inferencia no mesmo modulo por simbolo."""
    lock = runtime.get("model_lock")
    if lock is None:
        yield
        return
    with lock:
        yield


def resolve_dl_model_path(dl_config: dict, symbol: str) -> Path:
    """Resolve caminho do checkpoint PyTorch para um simbolo."""
    template = dl_config.get("model_path_template")
    if template:
        rel = str(template).format(symbol=symbol)
        return repo_path(rel).resolve()
    legacy = dl_config.get("model_path", "data/deep_learning_model.pth")
    return repo_path(legacy).resolve()


def granularity_seconds(orch) -> int:
    """Retorna granularidade OHLC em segundos."""
    return int(orch.config.get("data_handler", {}).get("granularity", 60))


def get_symbol_runtime(orch, symbol: str, dl_config: dict, params: dict) -> dict:
    """Carrega ou inicializa estado de modelo e normalizacao por simbolo."""
    if not hasattr(orch, "_dl_runtime"):
        orch._dl_runtime = {}
    if symbol not in orch._dl_runtime:
        path = resolve_dl_model_path(dl_config, symbol)
        loaded = load_model_checkpoint(path, params=params)
        calibrator = CalibratorState()
        lookback = int(params.get("lookback", 30))
        deploy_ok = False
        deploy_win_rate = 0.0
        session_trained = False
        checkpoint_granularity = 60
        if path.exists():
            try:
                payload = torch.load(path, map_location=torch.device("cpu"), weights_only=True)
                if isinstance(payload, dict) and "granularity" in payload:
                    checkpoint_granularity = int(payload["granularity"])
            except Exception as exc:
                logger.debug("DL: Nao foi possivel carregar a granularidade do checkpoint em %s: %s", path, exc)
        if loaded is not None:
            (
                model,
                norm_stats,
                last_epoch,
                calibrator,
                lookback,
                val_accuracy,
                val_brier,
                val_ece,
                deploy_ok,
                deploy_win_rate,
            ) = loaded
            if bool(dl_config.get("online_training", True)):
                session_trained = bool(deploy_ok) and float(val_brier) + 1e-9 < 0.99
            else:
                session_trained = float(val_brier) + 1e-9 < 0.99
            logger.debug("DL: Checkpoint carregado para %s em %s", symbol, path)
        else:
            model = create_direction_model(
                arch=params.get("arch", "tcn"),
                input_dim=FEATURE_DIM,
                tcn_channels=params.get("tcn_channels"),
                tcn_dropout=float(params.get("tcn_dropout", 0.2)),
                rnn_hidden_size=int(params.get("rnn_hidden_size", 64)),
                rnn_num_layers=int(params.get("rnn_num_layers", 2)),
                rnn_dropout=float(params.get("rnn_dropout", 0.2)),
            )
            norm_stats = fit_norm_stats(np.zeros((1, lookback, FEATURE_DIM), dtype=np.float32))
            last_epoch = 0
            val_accuracy = 0.0
            val_brier = 1.0
            val_ece = 1.0
        inference_device = resolve_torch_device(dl_config, kind="inference")
        place_model(model, inference_device)
        log_device_once(inference_device, context="inferencia")
        orch._dl_runtime[symbol] = {
            "model": model,
            "norm_stats": norm_stats,
            "last_candle_epoch": last_epoch,
            "val_accuracy": val_accuracy,
            "calibrator": calibrator,
            "val_brier": val_brier,
            "val_ece": val_ece,
            "lookback": lookback,
            "deploy_ok": deploy_ok,
            "deploy_win_rate": deploy_win_rate,
            "session_trained": session_trained,
            "model_lock": threading.RLock(),
            "trained_granularity": checkpoint_granularity,
        }
    elif "model_lock" not in orch._dl_runtime[symbol]:
        orch._dl_runtime[symbol]["model_lock"] = threading.RLock()
    return orch._dl_runtime[symbol]


def candle_epoch(orch, symbol: str) -> int:
    """Obtem epoch da ultima vela disponivel no stream."""
    getter = getattr(orch.stream, "get_last_candle_epoch", None)
    if callable(getter):
        epoch = getter(symbol)
        return int(epoch) if epoch is not None else 0
    return 0
