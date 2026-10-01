"""Funcoes de perda e penalizacoes anti-colapso 50/50 para treino TCN."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch
from torch import nn

from aether_paths import repo_path
from src.application.services.deep_learning.dl_device import tensor_from_numpy


def _read_dl_settings() -> dict[str, Any]:
    """Le o bloco deep_learning de settings.json."""
    path = repo_path("config", "settings.json")
    with path.open(encoding="utf-8") as handle:
        full = json.load(handle)
    dl = full.get("deep_learning") if isinstance(full, dict) else None
    return dl if isinstance(dl, dict) else {}


def aux_regression_weight() -> float:
    """Le aux_regression_weight de settings."""
    dl = _read_dl_settings()
    if "aux_regression_weight" not in dl:
        raise ValueError("deep_learning.aux_regression_weight obrigatorio")
    return float(dl["aux_regression_weight"])


def anti_collapse_loss_knobs() -> tuple[float, float, float]:
    """Retorna (margin_penalty_weight, entropy_penalty_weight, margin_floor)."""
    dl = _read_dl_settings()
    margin_w = float(dl.get("confidence_margin_penalty_weight", 0.15))
    entropy_w = float(dl.get("entropy_penalty_weight", 0.10))
    margin_floor = float(dl.get("confidence_margin_floor", 0.08))
    return margin_w, entropy_w, margin_floor


def model_core(model: Any) -> Any:
    """Extrai o modelo interno se ele for envelopado."""
    return getattr(model, "inner", model)


def calculate_masked_loss(
    model: Any,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    mask_batch: np.ndarray,
    weights: list[float],
    device: torch.device,
    *,
    label_smoothing: float,
    focal_gamma: float,
    delta_batch: np.ndarray | None = None,
    aux_regression_weight_val: float | None = None,
    confidence_margin_weight: float | None = None,
    entropy_penalty_weight: float | None = None,
    confidence_margin_floor: float | None = None,
) -> torch.Tensor:
    """Calcula perda de classificacao, margem de nitidez, entropia e regressao auxiliar."""
    smooth = max(0.0, min(0.2, float(label_smoothing)))
    targets = y_batch * (1.0 - smooth) + 0.5 * smooth
    core = model_core(model)
    x_tensor = tensor_from_numpy(x_batch, device)
    use_aux = delta_batch is not None and hasattr(core, "regression_head")
    if use_aux:
        try:
            logits, aux_pred = core(x_tensor, logits=True, return_aux=True)
        except TypeError:
            logits, aux_pred = core(x_tensor)
        logits = logits.squeeze(-1).clamp(-30.0, 30.0)
        aux_pred = aux_pred.squeeze(-1)
    else:
        try:
            logits = model(x_tensor, logits=True)
        except TypeError:
            logits = model(x_tensor)
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = logits.squeeze(-1).clamp(-30.0, 30.0)

    if aux_regression_weight_val is None:
        aux_regression_weight_val = aux_regression_weight()
    def_margin_w, def_ent_w, def_floor = anti_collapse_loss_knobs()
    m_weight = def_margin_w if confidence_margin_weight is None else float(confidence_margin_weight)
    e_weight = def_ent_w if entropy_penalty_weight is None else float(entropy_penalty_weight)
    m_floor = def_floor if confidence_margin_floor is None else float(confidence_margin_floor)

    target_t = tensor_from_numpy(targets, device).clamp(0.0, 1.0)
    mask_t = tensor_from_numpy(mask_batch, device)
    loss_vec = nn.functional.binary_cross_entropy_with_logits(logits, target_t, reduction="none")

    preds = torch.sigmoid(logits)
    if focal_gamma > 0.0:
        pt = torch.where(target_t >= 0.5, preds, 1.0 - preds)
        pos_frac = (target_t >= 0.5).float().mean().clamp(0.05, 0.95)
        alpha = torch.where(target_t >= 0.5, 1.0 - pos_frac, pos_frac)
        loss_vec = loss_vec * (2.0 * alpha) * torch.pow(2.0 * (1.0 - pt).clamp(min=1e-6), float(focal_gamma))

    if m_weight > 0.0:
        margin_dist = torch.abs(preds - 0.5)
        margin_penalty = torch.relu(m_floor - margin_dist).pow(2)
        loss_vec = loss_vec + float(m_weight) * margin_penalty

    if e_weight > 0.0:
        eps = 1e-6
        p_c = preds.clamp(eps, 1.0 - eps)
        entropy = -(p_c * torch.log(p_c) + (1.0 - p_c) * torch.log(1.0 - p_c))
        loss_vec = loss_vec + float(e_weight) * entropy

    w = tensor_from_numpy(np.asarray(weights, dtype=np.float32), device)
    weighted = loss_vec * mask_t * w
    denom = (mask_t * w).sum().clamp(min=1e-6)
    cls_loss = weighted.sum() / denom
    if not use_aux:
        return cls_loss
    delta_t = tensor_from_numpy(np.asarray(delta_batch, dtype=np.float32), device)
    reg_vec = nn.functional.mse_loss(aux_pred, delta_t, reduction="none")
    reg_weighted = (reg_vec * mask_t * w).sum() / denom
    return cls_loss + float(aux_regression_weight_val) * reg_weighted
