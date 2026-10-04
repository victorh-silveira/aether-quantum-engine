"""Loop de epocas e perda mascarada do treino walk-forward TCN."""

import json
import math

import numpy as np
import torch
from torch import optim

from aether_paths import repo_path
from src.application.services.deep_learning.dl_training_checkpoint import (
    checkpoint_if_improved,
    prefer_sharp_checkpoint,
    val_collapse_hit,
)
from src.application.services.deep_learning.dl_training_loss import (
    calculate_masked_loss as _masked_loss,
)


def _aux_regression_weight() -> float:
    """Le aux_regression_weight de settings."""
    path = repo_path("config", "settings.json")
    with path.open(encoding="utf-8") as handle:
        full = json.load(handle)
    dl = full.get("deep_learning") if isinstance(full, dict) else None
    if not isinstance(dl, dict) or "aux_regression_weight" not in dl:
        raise ValueError("deep_learning.aux_regression_weight obrigatorio")
    return float(dl["aux_regression_weight"])


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
    asymmetric_payout_loss: bool | None = None,
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
            asymmetric_payout_loss=asymmetric_payout_loss,
        )
        value = float(loss.item())
    model.train()
    return value if math.isfinite(value) else float("inf")


def _build_lr_scheduler(
    optimizer: optim.Optimizer, lr_scheduler: str, *, epochs: int, early_stopping_patience: int, lr: float
):
    """Instancia scheduler cosine ou reduce_on_plateau."""
    mode = str(lr_scheduler).strip().lower()
    if mode == "reduce_on_plateau":
        sched = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=max(2, early_stopping_patience // 3),
            min_lr=max(lr * 0.02, 1e-6),
        )
    else:
        sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=max(lr * 0.05, 1e-6))
    return mode, sched


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
    asymmetric_payout_loss: bool | None = None,
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
            asymmetric_payout_loss=asymmetric_payout_loss,
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
    min_epochs: int = 0,
    lr_scheduler: str = "cosine",
    progress_cb=None,
    delta_train: np.ndarray | None = None,
    min_oos_sharpness: float = 0.01,
    min_val_accuracy: float = 0.53,
    deploy_gate_cfg: dict | None = None,
    asymmetric_payout_loss: bool | None = None,
) -> tuple[float, None | dict, int]:
    """Executa epocas de treino com early stopping."""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=max(0.0, weight_decay))
    scheduler_mode, scheduler = _build_lr_scheduler(
        optimizer,
        lr_scheduler,
        epochs=max(1, epochs),
        early_stopping_patience=early_stopping_patience,
        lr=lr,
    )
    model.train()
    total_loss, patience_counter, epochs_ran = 0.0, 0, 0
    last_epoch_loss = 0.0
    best_state, best_sharp_state, best_fallback_state = None, None, None
    best_val_loss, best_sharp_loss, best_val_acc, best_sharp_acc = float("inf"), float("inf"), -1.0, -1.0
    best_fallback_acc = -1.0
    best_sharp_value = -1.0
    patience, min_ep, total_epochs = max(0, int(early_stopping_patience)), max(0, int(min_epochs)), max(1, epochs)
    weight_arr = np.asarray(weights, dtype=np.float32)
    sharp_floor, acc_floor = float(min_oos_sharpness), float(min_val_accuracy)
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
            asymmetric_payout_loss=asymmetric_payout_loss,
        )
        if batch_count == 0 or not math.isfinite(mean_epoch_loss):
            if best_state is not None:
                model.load_state_dict(best_state)
            continue
        total_loss += mean_epoch_loss
        last_epoch_loss = mean_epoch_loss
        val_loss = _validation_loss(
            model,
            x_val,
            y_val,
            mask_val,
            device,
            focal_gamma=0.0,
            asymmetric_payout_loss=asymmetric_payout_loss,
        )
        val_acc, val_sharp, collapse_hit = val_collapse_hit(model, x_val, y_val, mask_val, deploy_gate_cfg)
        model.train()
        if collapse_hit and val_acc > best_fallback_acc:
            best_fallback_acc = float(val_acc)
            best_fallback_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if progress_cb is not None:
            progress_cb(epochs_ran, total_epochs, mean_epoch_loss, float(val_acc))
        ck_res = checkpoint_if_improved(
            model,
            val_loss=val_loss,
            val_acc=float(val_acc),
            val_sharpness=float(val_sharp),
            min_sharpness=sharp_floor,
            min_val_accuracy=acc_floor,
            best_val_loss=best_val_loss,
            best_val_acc=best_val_acc,
            best_sharp_acc=best_sharp_acc,
            best_sharp_loss=best_sharp_loss,
            best_sharp_value=best_sharp_value,
            collapse_hit=bool(collapse_hit),
        )
        best_val_loss, best_val_acc, best_sharp_acc, best_sharp_loss = ck_res[:4]
        best_sharp_value, improved_state, sharp_state = ck_res[4:7]
        if improved_state is not None:
            best_state, patience_counter = improved_state, 0
        elif sharp_state is not None:
            patience_counter = 0
        elif epochs_ran >= min_ep:
            patience_counter += 1
            eff = patience
            if patience > 0 and patience_counter >= eff:
                break
        if sharp_state is not None:
            best_sharp_state = sharp_state
        scheduler.step(val_loss) if scheduler_mode == "reduce_on_plateau" else scheduler.step()
    chosen = prefer_sharp_checkpoint(
        best_state,
        best_sharp_state,
        best_acc=float(best_val_acc),
        sharp_acc=float(best_sharp_acc),
        min_val_accuracy=float(acc_floor),
    )
    if chosen is None:
        chosen = best_fallback_state
    final_loss = (
        last_epoch_loss if math.isfinite(last_epoch_loss) and last_epoch_loss > 0 else (total_loss / max(epochs_ran, 1))
    )
    return final_loss, chosen, epochs_ran
