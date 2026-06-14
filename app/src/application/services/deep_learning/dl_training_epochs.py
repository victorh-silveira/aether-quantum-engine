"""Loop de epocas e perda mascarada do treino walk-forward TCN."""

import math

import numpy as np
import torch
from torch import nn, optim

from src.application.services.deep_learning.dl_device import tensor_from_numpy
from src.application.services.deep_learning.model import model_accuracy


def _masked_loss(
    model,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    mask_batch: np.ndarray,
    weights: list[float],
    device: torch.device,
    *,
    label_smoothing: float,
    focal_gamma: float,
) -> torch.Tensor:
    """Calcula BCE mascarada com pesos e focal loss opcional."""
    smooth = max(0.0, min(0.2, float(label_smoothing)))
    targets = y_batch * (1.0 - smooth) + 0.5 * smooth
    logits = model(tensor_from_numpy(x_batch, device), logits=True).clamp(-30.0, 30.0)
    target_t = tensor_from_numpy(targets, device).clamp(0.0, 1.0)
    mask_t = tensor_from_numpy(mask_batch, device)
    loss_vec = nn.functional.binary_cross_entropy_with_logits(logits, target_t, reduction="none")
    if focal_gamma > 0.0:
        preds = torch.sigmoid(logits)
        pt = torch.where(target_t >= 0.5, preds, 1.0 - preds)
        loss_vec = loss_vec * torch.pow(1.0 - pt, float(focal_gamma))
    w = tensor_from_numpy(np.asarray(weights, dtype=np.float32), device)
    weighted = loss_vec * mask_t * w
    denom = (mask_t * w).sum().clamp(min=1e-6)
    return weighted.sum() / denom


def _shuffled_batch_indices(n: int, batch_size: int) -> list[np.ndarray]:
    """Gera indices em mini-lotes embaralhados para uma epoca."""
    size = max(1, int(batch_size))
    order = np.random.permutation(n)
    if size >= n:
        return [order]
    return [order[i : i + size] for i in range(0, n, size)]


def fit_training_epochs(
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    mask_train: np.ndarray,
    weights: list[float],
    x_val: np.ndarray,
    y_val: np.ndarray,
    mask_val: np.ndarray,
    device: torch.device,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    label_smoothing: float,
    focal_gamma: float,
    progress_cb=None,
) -> tuple[float, None | dict, int]:
    """Executa epocas de treino e guarda o melhor estado pela validacao."""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=max(0.0, weight_decay))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=max(lr * 0.05, 1e-6),
    )
    model.train()
    total_loss = 0.0
    best_state = None
    best_score = -1.0
    total_epochs = max(1, epochs)
    epochs_ran = 0
    weight_arr = np.asarray(weights, dtype=np.float32)
    for epoch_idx in range(total_epochs):
        epochs_ran = epoch_idx + 1
        epoch_loss = 0.0
        batch_count = 0
        for batch_idx in _shuffled_batch_indices(len(x_train), batch_size):
            optimizer.zero_grad()
            batch_w = weight_arr[batch_idx].tolist()
            loss = _masked_loss(
                model,
                x_train[batch_idx],
                y_train[batch_idx],
                mask_train[batch_idx],
                batch_w,
                device,
                label_smoothing=label_smoothing,
                focal_gamma=focal_gamma,
            )
            loss_value = float(loss.item())
            if not math.isfinite(loss_value):
                optimizer.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss_value
            batch_count += 1
        mean_epoch_loss = epoch_loss / max(batch_count, 1)
        if batch_count == 0 or not math.isfinite(mean_epoch_loss):
            if best_state is not None:
                model.load_state_dict(best_state)
            continue
        total_loss += mean_epoch_loss
        val_acc = model_accuracy(model, x_val, y_val, mask_val)
        if progress_cb is not None:
            progress_cb(epochs_ran, total_epochs, mean_epoch_loss, float(val_acc))
        with torch.no_grad():
            val_probs = model(tensor_from_numpy(x_val, device)).squeeze(-1).detach().cpu().numpy()
        val_brier = float(np.mean((val_probs - y_val) ** 2)) if len(y_val) else 1.0
        score = 0.6 * val_acc + 0.4 * (1.0 - val_brier)
        if score >= best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        scheduler.step()
    avg = total_loss / max(epochs_ran, 1)
    return avg, best_state, epochs_ran
