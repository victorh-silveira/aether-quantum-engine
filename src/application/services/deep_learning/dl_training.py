"""Treino walk-forward TCN com split purged e calibracao."""

import logging
import math

import numpy as np
import torch

from src.application.services.deep_learning.dl_calibration import apply_calibrator_stable
from src.application.services.deep_learning.dl_calibration_fit import (
    calibrator_entropy_metrics,
    fit_calibrator,
    maybe_identity_on_oos_collapse,
)
from src.application.services.deep_learning.dl_calibration_sharpen import (
    maybe_temperature_sharpen_for_export,
)
from src.application.services.deep_learning.dl_calibration_variance import (
    maybe_identity_on_variance_collapse,
)
from src.application.services.deep_learning.dl_device import (
    log_device_once,
    place_model,
    resolve_torch_device,
    tensor_from_numpy,
)
from src.application.services.deep_learning.dl_features import extract_sequences_and_deltas
from src.application.services.deep_learning.dl_preflight_benchmark import run_preflight_linear_baseline
from src.application.services.deep_learning.dl_sample_weighting import (
    compose_train_weights,
    label_call_fraction,
    minority_class_recall,
    parse_sample_weighting_config,
)
from src.application.services.deep_learning.dl_sharpness import mean_sharpness, resolve_calibration_sharpness_cfg
from src.application.services.deep_learning.dl_splits import purged_temporal_splits
from src.application.services.deep_learning.dl_training_epochs import fit_training_epochs
from src.application.services.deep_learning.model import (
    TrainResult,
    _model_raw_prob,
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
    symbol: str = "1HZ75V",
    open_: np.ndarray | None = None,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    micro: dict[str, np.ndarray] | None = None,
    batch_size: int = 128,
    dl_config: dict | None = None,
    progress_cb=None,
    asymmetric_payout_loss: bool | None = None,
) -> TrainResult | None:
    """Treina classificador com split purged e calibrador Platt."""
    device = resolve_torch_device(dl_config or {}, kind="training")
    place_model(model, device)
    log_device_once(device, context="treino")
    sparse_training = str(label_mode).lower() in ("triple_barrier", "quantum_multi_barrier")
    x_all, y_all, mask_all, delta_all = extract_sequences_and_deltas(
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
        filter_active=sparse_training,
    )
    split_count = len(x_all)
    effective_val_bars = min(int(validation_bars), max(5, int(split_count * 0.20)))
    splits = purged_temporal_splits(
        split_count,
        effective_val_bars,
        calib_ratio=calib_ratio,
        embargo=max(1, int(label_horizon_bars) + int(label_smooth_bars) - 1),
        stride=max(1, int(label_horizon_bars)),
    )
    if splits is None:
        logger.debug(
            "DL_TRAIN: amostras insuficientes para split (n=%d lookback=%d val=%d).",
            split_count,
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
    delta_train = delta_all[train_sl]
    acc_dummy, acc_linear = run_preflight_linear_baseline(x_train, y_train, x_val, y_val)
    quality_cfg = (dl_config.get("training_quality") or {}) if isinstance(dl_config, dict) else {}
    min_linear_acc = float(quality_cfg.get("min_linear_preflight_acc", 0.0))
    if min_linear_acc > 0.0 and acc_linear <= min_linear_acc:
        logger.warning(
            "DL PRE-FLIGHT BENCHMARK: acuracia de validacao %.1f%% <= piso %.1f%% — abortando treino TCN",
            acc_linear * 100.0,
            min_linear_acc * 100.0,
        )
        return None
    weighting_cfg = parse_sample_weighting_config(dl_config if isinstance(dl_config, dict) else None)
    weights = compose_train_weights(
        sample_weights,
        y_train,
        full_n=len(y_all),
        train_index=train_sl,
        weighting_cfg=weighting_cfg,
        deltas=delta_train,
    )
    label_call_frac = label_call_fraction(y_train)
    patience, min_epochs, label_smoothing, focal_gamma, lr_scheduler = 6, 0, 0.0, 0.0, "cosine"
    if dl_config is not None:
        patience = max(0, int(dl_config.get("early_stopping_patience", 6)))
        min_epochs = max(0, int(dl_config.get("min_epochs", 0)))
        label_smoothing = float(dl_config.get("label_smoothing", 0.0))
        focal_gamma = float(dl_config.get("focal_gamma", 0.0))
        lr_scheduler = str(dl_config.get("lr_scheduler") or "cosine").strip().lower() or "cosine"
    train_n = len(x_train)
    power = max(3, int(math.floor(math.log2(max(1, train_n / 4))))) if train_n > 0 else 3
    dynamic_cap = 32 if int(granularity) >= 86400 or train_n <= 120 else int(batch_size)
    effective_batch = max(8, min(dynamic_cap, 1 << power))
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
        batch_size=effective_batch,
        lr=lr,
        weight_decay=weight_decay,
        label_smoothing=label_smoothing,
        focal_gamma=focal_gamma,
        lr_scheduler=lr_scheduler,
        early_stopping_patience=patience,
        min_epochs=min_epochs,
        progress_cb=progress_cb,
        delta_train=delta_train,
        min_oos_sharpness=float(
            resolve_calibration_sharpness_cfg(
                (dl_config or {}).get("calibration") if isinstance(dl_config, dict) else None
            )["min_oos_sharpness"]
        ),
        min_val_accuracy=float(
            (((dl_config or {}).get("training_quality") or {}).get("soft_min_val_accuracy", 0.53))
            if isinstance(dl_config, dict)
            else 0.53
        ),
        deploy_gate_cfg=((dl_config or {}).get("training_quality") if isinstance(dl_config, dict) else None),
        asymmetric_payout_loss=asymmetric_payout_loss,
    )
    if best_state is not None:
        model.load_state_dict(best_state)
        place_model(model, device)
    val_accuracy = model_accuracy(model, x_val, y_val, mask_val)
    raw_val = _model_raw_prob(model, x_val) if len(x_val) else np.asarray([], dtype=np.float32)
    pred_call = np.asarray(raw_val >= 0.5, dtype=bool)
    active = np.ones(len(y_val), dtype=bool)
    if mask_val is not None and len(mask_val) == len(y_val):
        active = np.asarray(mask_val) > 0.5
    y_active = np.asarray(y_val)[active] if len(y_val) else np.asarray([])
    pred_active = pred_call[active] if pred_call.size == len(y_val) else pred_call
    minority_recall = minority_class_recall(y_active, pred_active)
    pred_call_frac = float(np.mean(pred_active)) if getattr(pred_active, "size", 0) else 0.5
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
    raw_probs = [float(p) for p in raw_calib]
    m_cal = mask_all[calib_sl] if mask_all is not None and len(mask_all) == len(x_all) else None
    act_c = [i for i, m in enumerate(m_cal) if float(m) > 0.5] if m_cal is not None else []
    p_fit = [raw_probs[i] for i in act_c] if len(act_c) >= 20 else raw_probs
    y_fit = [float(y_calib[i]) for i in act_c] if len(act_c) >= 20 else [float(y) for y in y_calib]
    calibrator = fit_calibrator(
        p_fit,
        y_fit,
        calibration_cfg=calibration_cfg if isinstance(calibration_cfg, dict) else None,
    )
    raw_sharpness = mean_sharpness(raw_probs)
    raw_val = _model_raw_prob(model, x_val) if len(x_val) else np.asarray([], dtype=np.float32)
    val_probs = [float(p) for p in raw_val]
    sharp_floor = float(
        resolve_calibration_sharpness_cfg(calibration_cfg if isinstance(calibration_cfg, dict) else None)[
            "min_oos_sharpness"
        ]
    )
    if val_probs:
        calibrator, oos_sharpness = maybe_identity_on_oos_collapse(
            calibrator,
            val_probs=val_probs,
            min_oos_sharpness=sharp_floor,
        )
        calibrator, oos_sharpness = maybe_temperature_sharpen_for_export(
            calibrator,
            val_probs=val_probs,
            min_oos_sharpness=sharp_floor,
        )
        calibrator = maybe_identity_on_variance_collapse(calibrator, probs=val_probs)
        raw_sharpness = mean_sharpness(val_probs)
    else:
        calibrated_holdout = [float(apply_calibrator_stable(float(p), calibrator)) for p in raw_probs]
        oos_sharpness = mean_sharpness(calibrated_holdout)
        calibrator = maybe_identity_on_variance_collapse(calibrator, probs=raw_probs)
    entropy_meta = calibrator_entropy_metrics(
        [float(p) for p in raw_calib],
        [float(y) for y in y_calib],
        calibrator,
        calibration_cfg=calibration_cfg if isinstance(calibration_cfg, dict) else None,
    )
    val_brier, val_ece = evaluate_calibrated_metrics(model, x_val, y_val, calibrator, mask=mask_val)
    train_accuracy = model_accuracy(model, x_train, y_train, mask_train)
    val_majority = max(float(np.mean(y_val)), 1.0 - float(np.mean(y_val)))
    logger.info(
        "DL TREINO DIAG | train_n=%d val_n=%d | train_acc=%.3f val_acc=%.3f "
        "gap=%.3f majority_val=%.3f | calibrator=%s sharpness=%.4f",
        len(x_train),
        len(x_val),
        train_accuracy,
        val_accuracy,
        train_accuracy - val_accuracy,
        val_majority,
        calibrator.method,
        oos_sharpness,
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
        oos_sharpness=float(oos_sharpness),
        raw_sharpness=float(raw_sharpness),
        label_call_frac=float(label_call_frac),
        pred_call_frac=float(pred_call_frac),
        minority_recall=float(minority_recall),
    )


def train_model_online(
    model, prices: np.ndarray, lookback: int, epochs: int, lr: float, validation_bars: int = 30, **kwargs
) -> float:
    """Wrapper compativel que retorna apenas a perda media do treino."""
    result = train_model_walkforward(model, prices, lookback, epochs, lr, validation_bars, **kwargs)
    return 0.0 if result is None else result.avg_loss
