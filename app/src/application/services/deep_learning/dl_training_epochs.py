"""Loop de epocas e perda mascarada do treino walk-forward TCN."""

import json
import math

import numpy as np
import torch
from torch import nn, optim

from aether_paths import repo_path
from src.application.services.deep_learning.dl_device import tensor_from_numpy
from src.application.services.deep_learning.model import model_accuracy


def _aux_regression_weight() -> float:
    """Le aux_regression_weight de settings."""
    path = repo_path("config", "settings.json")
    with path.open(encoding="utf-8") as handle:
        full = json.load(handle)
    dl = full.get("deep_learning") if isinstance(full, dict) else None
    if not isinstance(dl, dict) or "aux_regression_weight" not in dl:
        raise ValueError("deep_learning.aux_regression_weight obrigatorio")
    return float(dl["aux_regression_weight"])


def _model_core(model):
    """Extrai o modelo interno se ele for envelopado."""
    return getattr(model, "inner", model)


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
    delta_batch: np.ndarray | None = None,
    aux_regression_weight: float | None = None,
) -> torch.Tensor:
    """Calcula BCE mascarada com pesos, focal loss opcional e regressao auxiliar TCN."""
    smooth = max(0.0, min(0.2, float(label_smoothing)))
    targets = y_batch * (1.0 - smooth) + 0.5 * smooth
    core = _model_core(model)
    x_tensor = tensor_from_numpy(x_batch, device)
    use_aux = delta_batch is not None and hasattr(core, "regression_head")
    if use_aux:
        logits, aux_pred = core(x_tensor, logits=True, return_aux=True)
        logits = logits.clamp(-30.0, 30.0)
    else:
        logits = model(x_tensor, logits=True)
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = logits.clamp(-30.0, 30.0)
    if aux_regression_weight is None:
        aux_regression_weight = _aux_regression_weight()
    target_t = tensor_from_numpy(targets, device).clamp(0.0, 1.0)
    mask_t = tensor_from_numpy(mask_batch, device)
    loss_vec = nn.functional.binary_cross_entropy_with_logits(logits, target_t, reduction="none")
    if focal_gamma > 0.0:
        preds = torch.sigmoid(logits)
        pt = torch.where(target_t >= 0.5, preds, 1.0 - preds)
        loss_vec = loss_vec * torch.pow(2.0 * (1.0 - pt).clamp(min=1e-6), float(focal_gamma))
    w = tensor_from_numpy(np.asarray(weights, dtype=np.float32), device)
    weighted = loss_vec * mask_t * w
    denom = (mask_t * w).sum().clamp(min=1e-6)
    cls_loss = weighted.sum() / denom
    if not use_aux:
        return cls_loss
    delta_t = tensor_from_numpy(np.asarray(delta_batch, dtype=np.float32), device)
    reg_vec = nn.functional.mse_loss(aux_pred, delta_t, reduction="none")
    reg_weighted = (reg_vec * mask_t * w).sum() / denom
    return cls_loss + float(aux_regression_weight) * reg_weighted


def _shuffled_batch_indices(n: int, batch_size: int) -> list[np.ndarray]:
    """Gera indices em mini-lotes embaralhados para uma epoca."""
    size = max(1, int(batch_size))
    order = np.random.permutation(n)
    if size >= n:
        return [order]
    return [order[i : i + size] for i in range(0, n, size)]


def _validation_loss(
    model,
    x_val: np.ndarray,
    y_val: np.ndarray,
    mask_val: np.ndarray,
    device: torch.device,
    *,
    focal_gamma: float = 0.0,
) -> float:
    """Calcula perda de validacao mascarada sem gradiente."""
    model.eval()
    with torch.no_grad():
        weights = [1.0] * len(y_val)
        loss = _masked_loss(
            model,
            x_val,
            y_val,
            mask_val,
            weights,
            device,
            label_smoothing=0.0,
            focal_gamma=focal_gamma,
        )
        value = float(loss.item())
    model.train()
    return value if math.isfinite(value) else float("inf")


def _build_lr_scheduler(
    optimizer: optim.Optimizer,
    lr_scheduler: str,
    *,
    epochs: int,
    early_stopping_patience: int,
    lr: float,
):
    """Instancia scheduler cosine ou reduce_on_plateau."""
    scheduler_mode = str(lr_scheduler).strip().lower()
    if scheduler_mode == "reduce_on_plateau":
        return scheduler_mode, optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=max(2, early_stopping_patience // 3),
            min_lr=max(lr * 0.02, 1e-6),
        )
    return scheduler_mode, optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=max(lr * 0.05, 1e-6),
    )


def _mean_epoch_loss(
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    mask_train: np.ndarray,
    weight_arr: np.ndarray,
    device: torch.device,
    *,
    batch_size: int,
    label_smoothing: float,
    focal_gamma: float,
    optimizer: optim.Optimizer,
    delta_train: np.ndarray | None = None,
) -> tuple[float, int]:
    """Executa uma epoca completa e retorna loss media e contagem de batches."""
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
            delta_batch=None if delta_train is None else delta_train[batch_idx],
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
    return epoch_loss / max(batch_count, 1), batch_count


def _checkpoint_if_improved(
    model,
    *,
    val_loss: float,
    val_acc: float,
    best_val_loss: float,
    best_val_acc: float,
) -> tuple[float, float, dict | None, bool]:
    """Atualiza melhor checkpoint quando loss ou val_acc melhoram."""
    loss_improved = val_loss + 1e-9 < best_val_loss
    acc_improved = val_acc > best_val_acc + 1e-6
    if not loss_improved and not acc_improved:
        return best_val_loss, best_val_acc, None, False
    if loss_improved:
        best_val_loss = val_loss
    if acc_improved:
        best_val_acc = val_acc
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return best_val_loss, best_val_acc, best_state, True


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
    early_stopping_patience: int = 6,
    lr_scheduler: str = "cosine",
    progress_cb=None,
    delta_train: np.ndarray | None = None,
) -> tuple[float, None | dict, int]:
    """Executa epocas de treino com early stopping pela perda de validacao."""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=max(0.0, weight_decay))
    scheduler_mode, scheduler = _build_lr_scheduler(
        optimizer,
        lr_scheduler,
        epochs=max(1, epochs),
        early_stopping_patience=early_stopping_patience,
        lr=lr,
    )
    model.train()
    total_loss = 0.0
    best_state = None
    best_val_loss = float("inf")
    best_val_acc = -1.0
    patience = max(0, int(early_stopping_patience))
    patience_counter = 0
    total_epochs = max(1, epochs)
    epochs_ran = 0
    weight_arr = np.asarray(weights, dtype=np.float32)
    for epoch_idx in range(total_epochs):
        epochs_ran = epoch_idx + 1
        mean_epoch_loss, batch_count = _mean_epoch_loss(
            model,
            x_train,
            y_train,
            mask_train,
            weight_arr,
            device,
            batch_size=batch_size,
            label_smoothing=label_smoothing,
            focal_gamma=focal_gamma,
            optimizer=optimizer,
            delta_train=delta_train,
        )
        if batch_count == 0 or not math.isfinite(mean_epoch_loss):
            if best_state is not None:
                model.load_state_dict(best_state)
            continue
        total_loss += mean_epoch_loss
        val_acc = model_accuracy(model, x_val, y_val, mask_val)
        val_loss = _validation_loss(model, x_val, y_val, mask_val, device, focal_gamma=focal_gamma)
        if progress_cb is not None:
            progress_cb(epochs_ran, total_epochs, mean_epoch_loss, float(val_acc))
        best_val_loss, best_val_acc, improved_state, improved = _checkpoint_if_improved(
            model,
            val_loss=val_loss,
            val_acc=float(val_acc),
            best_val_loss=best_val_loss,
            best_val_acc=best_val_acc,
        )
        if improved:
            best_state = improved_state
            patience_counter = 0
        else:
            patience_counter += 1
            if patience > 0 and patience_counter >= patience:
                break
        if scheduler_mode == "reduce_on_plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()
    avg = total_loss / max(epochs_ran, 1)
    return avg, best_state, epochs_ran
