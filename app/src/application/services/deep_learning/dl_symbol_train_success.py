"""Persistencia de checkpoint e metricas apos treino walk-forward."""

import logging
import time

import numpy as np

from src.application.services.deep_learning.dl_calibration import CalibratorState
from src.application.services.deep_learning.dl_deploy import apply_deploy_to_runtime
from src.application.services.deep_learning.dl_deploy_eval import evaluate_mini_deploy
from src.application.services.deep_learning.dl_gate_config import resolve_deploy_ok
from src.application.services.deep_learning.dl_model_artifacts import schedule_model_upload
from src.application.services.deep_learning.dl_retrain import clear_force_retrain, reset_bars_since_train
from src.application.services.deep_learning.dl_symbol_runtime import resolve_dl_model_path
from src.application.services.deep_learning.model import save_model_checkpoint


logger = logging.getLogger("AETH")


def apply_successful_symbol_train(
    symbol: str,
    runtime: dict,
    train_result,
    *,
    orch,
    model,
    prices: np.ndarray,
    norm_stats,
    params: dict,
    dl_config: dict,
    gate_cfg: dict,
    candle_epoch_value: int,
    granularity: int,
    level: int,
    started: float,
    open_: np.ndarray | None = None,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    micro=None,
) -> tuple[object, float]:
    """Persiste checkpoint, deploy gate e metricas apos treino walk-forward valido."""
    runtime["norm_stats"] = train_result.norm_stats
    norm_stats = train_result.norm_stats
    runtime["val_accuracy"] = train_result.val_accuracy
    runtime["calibrator"] = train_result.calibrator or CalibratorState()
    runtime["val_brier"] = train_result.val_brier
    runtime["val_ece"] = train_result.val_ece
    runtime["calibrated_entropy"] = float(getattr(train_result, "calibrated_entropy", 0.0))
    runtime["entropy_violation"] = bool(getattr(train_result, "entropy_violation", False))
    train_loss = train_result.avg_loss
    runtime["last_candle_epoch"] = candle_epoch_value
    mini_ok, deploy_wr, mini_brier = evaluate_mini_deploy(
        orch,
        symbol,
        model,
        prices,
        norm_stats,
        runtime,
        params,
        gate_cfg=gate_cfg,
        open_=open_,
        high=high,
        low=low,
        micro=micro,
    )
    deploy_ok = resolve_deploy_ok(
        mini_ok=mini_ok,
        val_accuracy=float(train_result.val_accuracy),
        val_brier=float(train_result.val_brier),
        gate_cfg=gate_cfg,
    )
    apply_deploy_to_runtime(
        runtime,
        deploy_ok=deploy_ok,
        deploy_win_rate=deploy_wr,
        val_brier=mini_brier if mini_ok else float(train_result.val_brier),
    )
    path = resolve_dl_model_path(dl_config, symbol)
    save_model_checkpoint(
        path,
        model,
        norm_stats,
        candle_epoch_value,
        lookback=params["lookback"],
        calibrator=runtime["calibrator"],
        arch=params["arch"],
        val_accuracy=runtime["val_accuracy"],
        val_brier=runtime["val_brier"],
        val_ece=runtime["val_ece"],
        deploy_ok=runtime["deploy_ok"],
        deploy_win_rate=runtime["deploy_win_rate"],
        granularity=granularity,
    )
    schedule_model_upload(
        orch,
        symbol,
        path,
        arch=str(params["arch"]),
        metadata={
            "val_accuracy": runtime["val_accuracy"],
            "calibrated_entropy": runtime.get("calibrated_entropy"),
            "entropy_violation": runtime.get("entropy_violation"),
        },
    )
    runtime["session_trained"] = True
    clear_force_retrain(orch, symbol)
    reset_bars_since_train(orch, symbol)
    logger.log(
        level,
        "DL TREINO | %s | concluido em %.0fs | epocas=%d | loss=%.4f | val_acc=%.2f | brier=%.3f | deploy=%s",
        symbol,
        time.monotonic() - started,
        int(getattr(train_result, "epochs_ran", 0)),
        float(train_loss or 0.0),
        float(runtime.get("val_accuracy", 0.0)),
        float(runtime.get("val_brier", 1.0)),
        bool(runtime.get("deploy_ok", False)),
    )
    logger.log(level, "")
    return norm_stats, train_loss
