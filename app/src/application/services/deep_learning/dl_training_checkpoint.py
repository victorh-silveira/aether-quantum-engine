"""Selecao de checkpoint de treino TCN (val_acc + sharpness + val_loss)."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.application.services.deep_learning.dl_gate_config import _majority_collapse_hit
from src.application.services.deep_learning.dl_sample_weighting import (
    label_call_fraction,
    minority_class_recall,
)
from src.application.services.deep_learning.dl_sharpness import mean_sharpness
from src.application.services.deep_learning.model import _model_raw_prob, model_accuracy


MAX_STABLE_VAL_LOSS = 0.70


def _ce_stable(val_loss: float) -> bool:
    """True quando a CE de validacao esta abaixo do piso de chute aleatorio."""
    return float(val_loss) + 1e-9 < MAX_STABLE_VAL_LOSS


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
    collapse_hit: bool = False,
) -> tuple[float, float, float, float, dict | None, dict | None, bool]:
    """Atualiza picos estaveis (CE < 0.70); sharp prefere maior val_acc (loss desempata)."""
    loss_improved = val_loss + 1e-9 < best_val_loss
    stable = _ce_stable(val_loss)
    acc_improved = bool(stable) and (not bool(collapse_hit)) and val_acc > best_val_acc + 1e-6
    if loss_improved:
        best_val_loss = val_loss
    best_state = None
    best_sharp_state = None
    if acc_improved:
        best_val_acc = val_acc
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    sharp_ok = float(val_sharpness) + 1e-12 >= float(min_sharpness)
    acc_floor_ok = float(val_acc) + 1e-9 >= float(min_val_accuracy)
    if sharp_ok and acc_floor_ok and stable and not bool(collapse_hit):
        first_sharp = best_sharp_acc < 0.0
        acc_better = val_acc > best_sharp_acc + 1e-6
        same_acc_better_loss = abs(val_acc - best_sharp_acc) <= 1e-6 and float(val_loss) + 1e-9 < float(best_sharp_loss)
        if first_sharp or acc_better or same_acc_better_loss:
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
        best_state,
        best_sharp_state,
        improved_any,
    )


def prefer_sharp_checkpoint(
    best_state: dict | None,
    best_sharp_state: dict | None,
) -> dict | None:
    """Prefere o melhor estado que respeitou o piso de sharpness OOS."""
    if best_sharp_state is not None:
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
