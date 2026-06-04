"""Treino walk-forward TCN com split purged, early stopping e calibracao."""

import logging

import numpy as np
import torch
from torch import nn, optim

from src.application.services.deep_learning.dl_calibration import fit_calibrator
from src.application.services.deep_learning.dl_features import extract_sequences
from src.application.services.deep_learning.dl_splits import purged_temporal_splits
from src.application.services.deep_learning.model import (
    TrainResult,
    evaluate_calibrated_metrics,
    fit_norm_stats,
    model_accuracy,
    normalize_sequences,
)


logger = logging.getLogger("AETH")


def _masked_loss(
    model,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    mask_batch: np.ndarray,
    weights: list[float],
    *,
    label_smoothing: float,
    focal_gamma: float,
) -> torch.Tensor:
    """Calcula BCE mascarada com pesos e focal loss opcional."""
    smooth = max(0.0, min(0.2, float(label_smoothing)))
    targets = y_batch * (1.0 - smooth) + 0.5 * smooth
    preds = model(torch.tensor(x_batch)).squeeze(-1)
    target_t = torch.tensor(targets, dtype=torch.float32)
    mask_t = torch.tensor(mask_batch, dtype=torch.float32)
    loss_vec = nn.functional.binary_cross_entropy(preds, target_t, reduction="none")
    if focal_gamma > 0.0:
        pt = torch.where(target_t >= 0.5, preds, 1.0 - preds)
        loss_vec = loss_vec * torch.pow(1.0 - pt, float(focal_gamma))
    w = torch.tensor(weights, dtype=torch.float32)
    weighted = loss_vec * mask_t * w
    denom = (mask_t * w).sum().clamp(min=1e-6)
    return weighted.sum() / denom


def _fit_epochs(
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    mask_train: np.ndarray,
    weights: list[float],
    x_val: np.ndarray,
    y_val: np.ndarray,
    mask_val: np.ndarray,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    label_smoothing: float,
    focal_gamma: float,
    early_stopping_patience: int,
) -> tuple[float, None | dict]:
    """Executa epocas de treino com early stopping em val loss composto."""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=max(0.0, weight_decay))
    model.train()
    total_loss = 0.0
    best_state = None
    best_score = -1.0
    patience_left = max(1, int(early_stopping_patience))
    for _ in range(max(1, epochs)):
        optimizer.zero_grad()
        loss = _masked_loss(
            model,
            x_train,
            y_train,
            mask_train,
            weights,
            label_smoothing=label_smoothing,
            focal_gamma=focal_gamma,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.item())
        val_acc = model_accuracy(model, x_val, y_val, mask_val)
        val_probs = model(torch.tensor(x_val)).squeeze(-1).detach().numpy()
        val_brier = float(np.mean((val_probs - y_val) ** 2)) if len(y_val) else 1.0
        score = 0.6 * val_acc + 0.4 * (1.0 - val_brier)
        if score >= best_score:
            best_score = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = max(1, int(early_stopping_patience))
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    return total_loss / max(epochs, 1), best_state


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
    label_smoothing: float = 0.05,
    label_min_move_pct: float = 0.0002,
    early_stopping_patience: int = 3,
    focal_gamma: float = 0.0,
    calib_ratio: float = 0.15,
    granularity: int = 300,
    pair_prices: np.ndarray | None = None,
    require_pair_label: bool = False,
    sym_is_bull: bool = True,
) -> TrainResult | None:
    """Treina TCN com split purged, early stopping e calibrador Platt."""
    x_all, y_all, mask_all = extract_sequences(
        prices,
        lookback,
        label_min_move_pct=label_min_move_pct,
        granularity=granularity,
        pair_prices=pair_prices,
        require_pair_label=require_pair_label,
        sym_is_bull=sym_is_bull,
    )
    if require_pair_label and len(mask_all) > 0 and float(mask_all.mean()) < 0.08:
        logger.info("DL_TRAIN: pair label ativo em %.1f%%; retreino sem filtro de par.", 100.0 * float(mask_all.mean()))
        x_all, y_all, mask_all = extract_sequences(
            prices,
            lookback,
            label_min_move_pct=label_min_move_pct,
            granularity=granularity,
            pair_prices=pair_prices,
            require_pair_label=False,
            sym_is_bull=sym_is_bull,
        )
    splits = purged_temporal_splits(len(x_all), validation_bars, calib_ratio=calib_ratio)
    if splits is None:
        logger.info(
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
    weights = sample_weights if sample_weights and len(sample_weights) == len(y_train) else [1.0] * len(y_train)
    avg_loss, best_state = _fit_epochs(
        model,
        x_train,
        y_train,
        mask_train,
        weights,
        x_val,
        y_val,
        mask_val,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        label_smoothing=label_smoothing,
        focal_gamma=focal_gamma,
        early_stopping_patience=early_stopping_patience,
    )
    if best_state is not None:
        model.load_state_dict(best_state)
    val_accuracy = model_accuracy(model, x_val, y_val, mask_val)
    model.eval()
    raw_calib = model(torch.tensor(normalize_sequences(x_all[calib_sl], norm_stats))).squeeze(-1).detach().numpy()
    calibrator = fit_calibrator([float(p) for p in raw_calib], [float(y) for y in y_calib])
    val_brier, val_ece = evaluate_calibrated_metrics(model, x_val, y_val, calibrator)
    logger.debug(
        "DL_TRAIN: loss=%.4f val_acc=%.3f brier=%.3f ece=%.3f samples=%d",
        avg_loss,
        val_accuracy,
        val_brier,
        val_ece,
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
