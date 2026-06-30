"""Tensores de stress para forward pass de sanidade TorchScript."""

from __future__ import annotations

import torch


def build_sanity_probe_tensors(
    lookback: int,
    feature_dim: int,
) -> list[tuple[str, torch.Tensor]]:
    """Monta batch de probes incluindo Z-scores extremos por feature."""
    lb = int(lookback)
    fd = int(feature_dim)
    zeros = torch.zeros(1, lb, fd, dtype=torch.float32)
    ones = torch.ones(1, lb, fd, dtype=torch.float32)
    pos_extreme = torch.full((1, lb, fd), 4.0, dtype=torch.float32)
    neg_extreme = torch.full((1, lb, fd), -4.0, dtype=torch.float32)
    mixed = torch.zeros(1, lb, fd, dtype=torch.float32)
    for feat in range(fd):
        mixed[:, :, feat] = 3.0 if feat % 2 == 0 else -3.0
    return [
        ("zeros", zeros),
        ("unit", ones),
        ("pos_extreme", pos_extreme),
        ("neg_extreme", neg_extreme),
        ("mixed", mixed),
    ]
