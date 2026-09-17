"""Selecao de checkpoint de treino TCN (val_acc + sharpness + val_loss)."""

from typing import Any

import numpy as np

from src.application.services.deep_learning.dl_gate_config import _majority_collapse_hit
from src.application.services.deep_learning.dl_sample_weighting import (
    label_call_fraction,
    minority_class_recall,
)
from src.application.services.deep_learning.dl_sharpness import mean_sharpness
from src.application.services.deep_learning.model import _model_raw_prob, model_accuracy


def checkpoint_if_improved(
    model,
    *,
    val_loss: float,
    val_acc: float,
    val_sharpness: float,
    min_sharpness: float,
    min_val_accuracy: float,
    best_val_loss: float,
    best_val_acc: float,
    best_sharp_acc: float,
    best_sharp_loss: float,
    best_sharp_value: float = -1.0,
    collapse_hit: bool = False,
) -> tuple[float, float, float, float, float, dict | None, dict | None, bool]:
    """Atualiza pico por maior val_acc; ramo sharp maximiza nitidez com ACC no piso.

    O piso de export (`min_sharpness`) nao bloqueia o ramo sharp no treino — so o gate
    final de export exige o piso. Collapse so bloqueia o ramo sharp.
    """
    _ = float(min_sharpness)
    loss_improved = val_loss + 1e-9 < best_val_loss
    if loss_improved:
        best_val_loss = val_loss
    best_state = None
    best_sharp_state = None
    acc_improved = val_acc > best_val_acc + 1e-6
    if acc_improved:
        best_val_acc = val_acc
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    acc_floor_ok = float(val_acc) + 1e-9 >= float(min_val_accuracy)
    if acc_floor_ok and not bool(collapse_hit):
        sharp_v = float(val_sharpness)
        first_sharp = best_sharp_value < 0.0
        sharper = sharp_v > float(best_sharp_value) + 1e-6
        same_sharp_better_acc = abs(sharp_v - float(best_sharp_value)) <= 1e-6 and val_acc > best_sharp_acc + 1e-6
        same_sharp_same_acc_better_loss = (
            abs(sharp_v - float(best_sharp_value)) <= 1e-6
            and abs(val_acc - best_sharp_acc) <= 1e-6
            and float(val_loss) + 1e-9 < float(best_sharp_loss)
        )
        if first_sharp or sharper or same_sharp_better_acc or same_sharp_same_acc_better_loss:
            best_sharp_value = sharp_v
            best_sharp_acc = val_acc
            best_sharp_loss = float(val_loss)
            best_sharp_state = (
                best_state
                if best_state is not None
                else {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            )
    improved_any = loss_improved or acc_improved
    return (
        best_val_loss,
        best_val_acc,
        best_sharp_acc,
        best_sharp_loss,
        best_sharp_value,
        best_state,
        best_sharp_state,
        improved_any,
    )


def prefer_sharp_checkpoint(
    best_state: dict | None,
    best_sharp_state: dict | None,
    *,
    best_acc: float = -1.0,
    sharp_acc: float = -1.0,
    min_val_accuracy: float = 0.53,
) -> dict | None:
    """Prefere sharp se a ACC sharp limpa o piso soft (export ainda exige nitidez)."""
    if best_sharp_state is None:
        return best_state
    if best_state is None:
        return best_sharp_state
    if float(sharp_acc) + 1e-9 >= float(min_val_accuracy):
        return best_sharp_state
    if float(sharp_acc) + 1e-9 >= float(best_acc) - 1e-6:
        return best_sharp_state
    return best_state


def val_collapse_hit(
    model,
    x_val: np.ndarray,
    y_val: np.ndarray,
    mask_val: np.ndarray | None,
    gate_cfg: dict[str, Any] | None,
) -> tuple[float, float, bool]:
    """Retorna (val_acc, val_sharpness, collapse_hit) no holdout."""
    val_acc = float(model_accuracy(model, x_val, y_val, mask_val))
    raw_val = _model_raw_prob(model, x_val) if len(x_val) else np.asarray([], dtype=np.float32)
    val_sharp = mean_sharpness([float(p) for p in raw_val]) if len(raw_val) else 0.0
    cfg = gate_cfg if isinstance(gate_cfg, dict) else {}
    pred_call = np.asarray(raw_val >= 0.5, dtype=bool) if len(raw_val) else np.asarray([], dtype=bool)
    active = np.ones(len(y_val), dtype=bool)
    if mask_val is not None and len(mask_val) == len(y_val):
        active = np.asarray(mask_val) > 0.5
    y_active = np.asarray(y_val)[active] if len(y_val) else np.asarray([])
    pred_active = pred_call[active] if pred_call.size == len(y_val) else pred_call
    hit = _majority_collapse_hit(
        cfg,
        label_call_frac=label_call_fraction(y_active) if y_active.size else None,
        pred_call_frac=float(np.mean(pred_active)) if getattr(pred_active, "size", 0) else None,
        minority_recall=minority_class_recall(y_active, pred_active) if y_active.size else None,
    )
    return val_acc, float(val_sharp), bool(hit)
