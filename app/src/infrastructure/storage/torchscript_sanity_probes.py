"""Tensores de stress para forward pass de sanidade TorchScript e Triton."""

from __future__ import annotations

import numpy as np
import torch


FEATURE_RSI_IDX = 5
FEATURE_CMO_IDX = 25
FEATURE_VOL_RATIO_IDX = 31

STRESSED_RSI = 0.99
STRESSED_CMO = 1.0
STRESSED_VOL_RATIO = 1.80


def build_stressed_regime_probe_tensor(lookback: int, feature_dim: int) -> torch.Tensor:
    """Monta tensor com regime estressado: RSI alto, CMO saturado e vol_ratio em expansao."""
    lb = int(lookback)
    fd = int(feature_dim)
    tensor = torch.zeros(1, lb, fd, dtype=torch.float32)
    tensor[:, :, FEATURE_RSI_IDX] = STRESSED_RSI
    tensor[:, :, FEATURE_CMO_IDX] = STRESSED_CMO
    tensor[:, :, FEATURE_VOL_RATIO_IDX] = STRESSED_VOL_RATIO
    return tensor


def build_stressed_regime_probe_ndarray(lookback: int, feature_dim: int) -> np.ndarray:
    """Versao NumPy do probe estressado para inferencia Triton."""
    return build_stressed_regime_probe_tensor(lookback, feature_dim).numpy()


def build_sanity_probe_tensors(
    lookback: int,
    feature_dim: int,
) -> list[tuple[str, torch.Tensor]]:
    """Monta batch de probes incluindo regime estressado por feature."""
    lb = int(lookback)
    fd = int(feature_dim)
    zeros = torch.zeros(1, lb, fd, dtype=torch.float32)
    ones = torch.ones(1, lb, fd, dtype=torch.float32)
    pos_extreme = torch.full((1, lb, fd), 4.0, dtype=torch.float32)
    neg_extreme = torch.full((1, lb, fd), -4.0, dtype=torch.float32)
    mixed = torch.zeros(1, lb, fd, dtype=torch.float32)
    for feat in range(fd):
        mixed[:, :, feat] = 3.0 if feat % 2 == 0 else -3.0
    stressed = build_stressed_regime_probe_tensor(lb, fd)
    return [
        ("zeros", zeros),
        ("unit", ones),
        ("pos_extreme", pos_extreme),
        ("neg_extreme", neg_extreme),
        ("mixed", mixed),
        ("stressed_regime", stressed),
    ]
