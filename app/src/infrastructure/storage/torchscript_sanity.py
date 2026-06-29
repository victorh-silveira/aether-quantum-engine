"""Forward pass de validacao para artefatos TorchScript."""

from __future__ import annotations

from pathlib import Path

import torch


def verify_torchscript_artifact(
    path: Path,
    *,
    lookback: int,
    feature_dim: int,
) -> None:
    """Falha se o TorchScript nao aceitar tensor dummy ou retornar NaN."""
    if not path.is_file():
        raise RuntimeError(f"TorchScript ausente: {path}")
    model = torch.jit.load(str(path), map_location=torch.device("cpu"))
    model.eval()
    dummy = torch.zeros(1, int(lookback), int(feature_dim), dtype=torch.float32)
    with torch.no_grad():
        out = model(dummy)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if not torch.is_tensor(out):
        raise RuntimeError(f"TorchScript retornou tipo invalido: {type(out).__name__}")
    if out.ndim < 1 or out.shape[0] < 1:
        raise RuntimeError(f"TorchScript shape invalida: {tuple(out.shape)}")
    if torch.isnan(out).any().item() or torch.isinf(out).any().item():
        raise RuntimeError("TorchScript produziu NaN ou Inf no forward de sanidade")
