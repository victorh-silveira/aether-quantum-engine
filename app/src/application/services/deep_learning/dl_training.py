"""Treino walk-forward TCN com split purged e calibracao."""

import logging

import numpy as np
import torch

from src.application.services.deep_learning.dl_calibration_fit import calibrator_entropy_metrics, fit_calibrator
from src.application.services.deep_learning.dl_device import (
    device_label,
    log_device_once,
    place_model,
    resolve_torch_device,
    tensor_from_numpy,
)
from src.application.services.deep_learning.dl_features import extract_sequences
from src.application.services.deep_learning.dl_sequence_extract import sequence_price_deltas
from src.application.services.deep_learning.dl_splits import purged_temporal_splits
from src.application.services.deep_learning.dl_training_epochs import fit_training_epochs
from src.application.services.deep_learning.model import (
    TrainResult,
    evaluate_calibrated_metrics,
    fit_norm_stats,
    model_accuracy,
    normalize_sequences,
)


logger = logging.getLogger("AETH")


def train_model_walkforward(
    model,
    prices: np.ndarray,
    lookback: int,
    epochs: int,
    lr: float,
    validation_bars: int,
    *,
    sample_weights: list[float] | None = None,
    weight_decay: float = 0.0,
    calib_ratio: float = 0.15,
    granularity: int = 60,
    label_horizon_bars: int = 1,
    label_smooth_bars: int = 1,
    label_mode: str = "ma_trend",
    label_ma_window: int = 5,
    implied_vol_bars: int = 60,
    symbol: str = "RDBULL",
    open_: np.ndarray | None = None,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    micro: dict[str, np.ndarray] | None = None,
    batch_size: int = 128,
    dl_config: dict | None = None,
    progress_cb=None,
) -> TrainResult | None:
    """Treina classificador com split purged e calibrador Platt."""
    device = resolve_torch_device(dl_config or {}, kind="training")
    place_model(model, device)
    log_device_once(device, context="treino")
    x_all, y_all, mask_all = extract_sequences(
        prices,
        lookback,
        granularity=granularity,
        label_horizon_bars=label_horizon_bars,
        label_smooth_bars=label_smooth_bars,
        label_mode=label_mode,
        label_ma_window=label_ma_window,
        implied_vol_bars=implied_vol_bars,
        symbol=symbol,
        open_=open_,
        high=high,
        low=low,
        micro=micro,
    )
    splits = purged_temporal_splits(
        len(x_all),
        validation_bars,
        calib_ratio=calib_ratio,
        embargo=label_horizon_bars,
    )
    if splits is None:
        logger.debug(
            "DL_TRAIN: amostras insuficientes para split (n=%d lookback=%d val=%d).",
            len(x_all),
            lookback,
            validation_bars,
        )
        return None
    train_sl, val_sl, calib_sl = splits
    norm_stats = fit_norm_stats(x_all[train_sl])
    x_train = normalize_sequences(x_all[train_sl], norm_stats)
    x_val = normalize_sequences(x_all[val_sl], norm_stats)
    y_train, mask_train = y_all[train_sl], mask_all[train_sl]
    y_val, mask_val = y_all[val_sl], mask_all[val_sl]
    y_calib = y_all[calib_sl]
    delta_all = sequence_price_deltas(
        prices,
        lookback,
        label_horizon_bars=label_horizon_bars,
        label_smooth_bars=label_smooth_bars,
        label_mode=label_mode,
        label_ma_window=label_ma_window,
    )
    delta_train = delta_all[train_sl] if len(delta_all) == len(x_all) else None
    weights = sample_weights if sample_weights and len(sample_weights) == len(y_train) else [1.0] * len(y_train)
    patience = 6
    label_smoothing = 0.0
    focal_gamma = 0.0
    lr_scheduler = "cosine"
    if dl_config is not None:
        patience = max(0, int(dl_config.get("early_stopping_patience", 6)))
        label_smoothing = float(dl_config.get("label_smoothing", 0.0))
        focal_gamma = float(dl_config.get("focal_gamma", 0.0))
        lr_scheduler = str(dl_config.get("lr_scheduler", "cosine")).strip().lower()
    avg_loss, best_state, epochs_ran = fit_training_epochs(
        model,
        x_train,
        y_train,
        mask_train,
        weights,
        x_val,
        y_val,
        mask_val,
        device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        label_smoothing=label_smoothing,
        focal_gamma=focal_gamma,
        lr_scheduler=lr_scheduler,
        early_stopping_patience=patience,
        progress_cb=progress_cb,
        delta_train=delta_train,
    )
    if best_state is not None:
        model.load_state_dict(best_state)
        place_model(model, device)
    val_accuracy = model_accuracy(model, x_val, y_val, mask_val)
    model.eval()
    with torch.no_grad():
        raw_calib = (
            model(tensor_from_numpy(normalize_sequences(x_all[calib_sl], norm_stats), device))
            .squeeze(-1)
            .detach()
            .cpu()
            .numpy()
        )
    calibration_cfg = (dl_config or {}).get("calibration") if isinstance(dl_config, dict) else None
    calibrator = fit_calibrator(
        [float(p) for p in raw_calib],
        [float(y) for y in y_calib],
        calibration_cfg=calibration_cfg if isinstance(calibration_cfg, dict) else None,
    )
    entropy_meta = calibrator_entropy_metrics(
        [float(p) for p in raw_calib],
        [float(y) for y in y_calib],
        calibrator,
        calibration_cfg=calibration_cfg if isinstance(calibration_cfg, dict) else None,
    )
    val_brier, val_ece = evaluate_calibrated_metrics(model, x_val, y_val, calibrator)
    logger.debug(
        "DL_TRAIN: device=%s batch=%d epocas=%d/%d loss=%.4f val_acc=%.3f brier=%.3f ece=%.3f method=%s samples=%d",
        device_label(device),
        max(1, int(batch_size)),
        epochs_ran,
        max(1, epochs),
        avg_loss,
        val_accuracy,
        val_brier,
        val_ece,
        calibrator.method,
        len(x_all),
    )
    return TrainResult(
        avg_loss=avg_loss,
        val_accuracy=val_accuracy,
        norm_stats=norm_stats,
        temperature=calibrator.temperature,
        calibrator=calibrator,
        val_brier=val_brier,
        val_ece=val_ece,
        epochs_ran=epochs_ran,
        calibrated_entropy=float(entropy_meta.get("calibrated_entropy", 0.0)),
        entropy_violation=bool(entropy_meta.get("entropy_violation", False)),
    )


def train_model_online(
    model,
    prices: np.ndarray,
    lookback: int,
    epochs: int,
    lr: float,
    validation_bars: int = 30,
    **kwargs,
) -> float:
    """Wrapper compativel que retorna apenas a perda media do treino."""
    result = train_model_walkforward(model, prices, lookback, epochs, lr, validation_bars, **kwargs)
    if result is None:
        return 0.0
    return result.avg_loss
