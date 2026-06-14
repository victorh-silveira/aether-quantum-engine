"""Persistencia de checkpoints PyTorch e exportacao TorchScript."""

import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.application.services.deep_learning.dl_calibration import (
    CalibratorState,
    calibrator_from_dict,
    calibrator_to_dict,
)
from src.application.services.deep_learning.dl_features import FEATURE_DIM
from src.application.services.deep_learning.dl_model_factory import create_direction_model
from src.application.services.deep_learning.dl_model_types import CHECKPOINT_VERSION, DEFAULT_ARCH, FeatureNormStats


logger = logging.getLogger("AETH")


def _scripted_path(path: Path) -> Path:
    """Retorna caminho do artefato TorchScript associado ao checkpoint."""
    return path.with_name(path.stem + "_ts.pt")


def _save_torchscript(model: nn.Module, path: Path, *, lookback: int) -> None:
    """Exporta TorchScript para inferencia rapida."""
    try:
        model.eval()
        device = next(model.parameters()).device
        example = torch.zeros(1, int(lookback), FEATURE_DIM, dtype=torch.float32, device=device)
        traced = torch.jit.trace(model, example, strict=False)
        traced.save(str(_scripted_path(path)))
    except Exception as exc:
        logger.debug("DL: TorchScript nao exportado para %s: %s", path, exc)


def save_model_checkpoint(
    path: Path,
    model: nn.Module,
    norm_stats: FeatureNormStats,
    last_candle_epoch: int,
    *,
    lookback: int,
    calibrator: CalibratorState | None = None,
    arch: str | None = None,
    val_accuracy: float | None = None,
    val_brier: float | None = None,
    val_ece: float | None = None,
    deploy_ok: bool | None = None,
    deploy_win_rate: float | None = None,
    granularity: int | None = None,
) -> None:
    """Persiste checkpoint com calibrador e metadados de arquitetura."""
    arch = arch or DEFAULT_ARCH
    path.parent.mkdir(parents=True, exist_ok=True)
    cal = calibrator or CalibratorState()
    payload = {
        "version": CHECKPOINT_VERSION,
        "arch": arch,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "norm_mean": norm_stats.mean,
        "norm_std": norm_stats.std,
        "feature_dim": FEATURE_DIM,
        "lookback": int(lookback),
        "last_candle_epoch": last_candle_epoch,
        "calibrator": calibrator_to_dict(cal),
    }
    if val_accuracy is not None:
        payload["val_accuracy"] = float(val_accuracy)
    if val_brier is not None:
        payload["val_brier"] = float(val_brier)
    if val_ece is not None:
        payload["val_ece"] = float(val_ece)
    if deploy_ok is not None:
        payload["deploy_ok"] = bool(deploy_ok)
    if deploy_win_rate is not None:
        payload["deploy_win_rate"] = float(deploy_win_rate)
    if granularity is not None:
        payload["granularity"] = int(granularity)
    torch.save(payload, path)
    _save_torchscript(model, path, lookback=lookback)


def load_model_checkpoint(
    path: Path,
    *,
    params: dict | None = None,
) -> tuple[nn.Module, FeatureNormStats, int, CalibratorState, int, float, float, float, bool, float] | None:
    """Carrega checkpoint ou descarta formatos incompativeis."""
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location=torch.device("cpu"), weights_only=False)  # nosec B614
    except Exception:
        logger.debug("DL: Checkpoint corrompido em %s; sera reiniciado.", path)
        return None
    if not isinstance(payload, dict) or "state_dict" not in payload:
        return None
    feature_dim = int(payload.get("feature_dim", payload.get("input_dim", FEATURE_DIM)))
    if feature_dim != FEATURE_DIM:
        logger.debug("DL: FEATURE_DIM incompativel em %s; sera reiniciado.", path)
        return None
    arch = str(payload.get("arch", DEFAULT_ARCH))
    cfg = params or {}
    model = create_direction_model(
        arch=arch,
        input_dim=feature_dim,
        tcn_channels=cfg.get("tcn_channels"),
        tcn_dropout=float(cfg.get("tcn_dropout", 0.2)),
        rnn_hidden_size=int(cfg.get("rnn_hidden_size", 64)),
        rnn_num_layers=int(cfg.get("rnn_num_layers", 2)),
        rnn_dropout=float(cfg.get("rnn_dropout", 0.2)),
    )
    try:
        model.load_state_dict(payload["state_dict"])
    except RuntimeError:
        logger.debug("DL: state_dict incompativel em %s; sera reiniciado.", path)
        return None
    scripted = _scripted_path(path)
    if scripted.exists():
        try:
            model = torch.jit.load(str(scripted), map_location=torch.device("cpu"))
        except Exception:
            logger.debug("DL: TorchScript invalido em %s; usando eager.", scripted)
    norm_stats = FeatureNormStats(
        mean=np.asarray(payload["norm_mean"], dtype=np.float32),
        std=np.asarray(payload["norm_std"], dtype=np.float32),
    )
    epoch = int(payload.get("last_candle_epoch", 0))
    calibrator = calibrator_from_dict(payload.get("calibrator"))
    lookback = int(payload.get("lookback", 30))
    val_accuracy = float(payload.get("val_accuracy", 0.0))
    val_brier = float(payload.get("val_brier", 1.0))
    val_ece = float(payload.get("val_ece", 1.0))
    deploy_ok = bool(payload.get("deploy_ok", False))
    deploy_win_rate = float(payload.get("deploy_win_rate", 0.0))
    return model, norm_stats, epoch, calibrator, lookback, val_accuracy, val_brier, val_ece, deploy_ok, deploy_win_rate
