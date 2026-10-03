"""Runtime de modelo e checkpoints por simbolo."""

import hashlib
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
from src.application.services.deep_learning.dl_params import resolve_dl_granularity
from src.application.services.deep_learning.model import (
    create_direction_model,
    fit_norm_stats,
    load_model_checkpoint,
)


logger = logging.getLogger("AETH")


def checkpoint_fingerprint(path: Path) -> str | None:
    """Identifica os bytes exatos do checkpoint usado na inferencia."""
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as checkpoint:
            for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


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
    """Retorna granularidade OHLC usada pelo treino/inferencia DL."""
    data = orch.config.get("data_handler", {}) if getattr(orch, "config", None) else {}
    dl = orch.config.get("deep_learning", {}) if getattr(orch, "config", None) else {}
    return resolve_dl_granularity(dl if isinstance(dl, dict) else {}, data if isinstance(data, dict) else {})


def get_symbol_runtime(orch, symbol: str, dl_config: dict, params: dict) -> dict:
    """Carrega ou inicializa estado de modelo e normalizacao por simbolo."""
    if not hasattr(orch, "_dl_runtime"):
        orch._dl_runtime = {}
    if symbol not in orch._dl_runtime:
        path = resolve_dl_model_path(dl_config, symbol)
        expected_lookback = int(params.get("lookback", 30))
        expected_granularity = granularity_seconds(orch)
        loaded = load_model_checkpoint(path, params=params)
        calibrator = CalibratorState()
        lookback = expected_lookback
        deploy_ok = False
        deploy_provisional_ok = False
        deploy_win_rate = 0.0
        session_trained = False
        checkpoint_granularity = expected_granularity
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
                ckpt_lookback,
                val_accuracy,
                val_brier,
                val_ece,
                deploy_ok,
                deploy_win_rate,
            ) = loaded
            if int(ckpt_lookback) != expected_lookback or int(checkpoint_granularity) != int(expected_granularity):
                if bool(dl_config.get("online_training", False)):
                    logger.info(
                        "DL: Checkpoint %s incompativel (lb=%s/%s gran=%s/%s); retreino.",
                        symbol,
                        ckpt_lookback,
                        expected_lookback,
                        checkpoint_granularity,
                        expected_granularity,
                    )
                else:
                    logger.info(
                        "DL: Checkpoint %s incompativel (lb=%s/%s gran=%s/%s); "
                        "online_training=off — rode app/scripts/batch/launch-train.bat "
                        "(treino separado do DEMO).",
                        symbol,
                        ckpt_lookback,
                        expected_lookback,
                        checkpoint_granularity,
                        expected_granularity,
                    )
                loaded = None
                checkpoint_granularity = expected_granularity
        if loaded is not None:
            lookback = int(ckpt_lookback)
            session_trained = True
            deploy_ok = False
            deploy_provisional_ok = False
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
            deploy_ok = False
            deploy_provisional_ok = False
            deploy_win_rate = 0.0
            session_trained = False
            calibrator = CalibratorState()
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
            "checkpoint_loaded": loaded is not None,
            "model_version": checkpoint_fingerprint(path) if loaded is not None else None,
            "deploy_provisional_ok": deploy_provisional_ok,
            "deploy_win_rate": deploy_win_rate,
            "session_trained": session_trained,
            "model_lock": threading.RLock(),
            "trained_granularity": checkpoint_granularity,
        }
    elif "model_lock" not in orch._dl_runtime[symbol]:
        orch._dl_runtime[symbol]["model_lock"] = threading.RLock()
    return orch._dl_runtime[symbol]


def candle_epoch(orch, symbol: str, *, timeframe: str | None = None) -> int:
    """Obtem epoch da ultima vela do timeframe de treino (macro/micro)."""
    stream = getattr(orch, "stream", None)
    if stream is None:
        return 0
    tf = str(timeframe or "").strip().lower()
    if tf == "micro":
        getter = getattr(stream, "get_last_micro_candle_epoch", None)
        if callable(getter):
            epoch = getter(symbol)
            return int(epoch) if epoch is not None else 0
    getter = getattr(stream, "get_last_candle_epoch", None)
    if callable(getter):
        epoch = getter(symbol)
        return int(epoch) if epoch is not None else 0
    return 0
