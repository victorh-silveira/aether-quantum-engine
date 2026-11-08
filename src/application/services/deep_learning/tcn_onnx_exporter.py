"""Exportador offline de modelos TCN PyTorch para o formato ONNX."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from src.application.services.deep_learning.dl_features import FEATURE_DIM
from src.application.services.deep_learning.dl_model_checkpoint import load_model_checkpoint


logger = logging.getLogger("AETH")


def export_tcn_to_onnx(
    model: torch.nn.Module,
    output_path: Path | str,
    *,
    lookback: int = 32,
    feature_dim: int = FEATURE_DIM,
    opset_version: int = 17,
) -> bool:
    """Exporta modelo PyTorch para grafo computacional ONNX estatico ou dinamico."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.zeros(1, int(lookback), int(feature_dim), dtype=torch.float32)
    model.eval()

    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(out_file),
            export_params=True,
            opset_version=int(opset_version),
            do_constant_folding=True,
            input_names=["input_features"],
            output_names=["logits"],
            dynamic_axes={
                "input_features": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
        )
        logger.info("ONNX: Modelo exportado com sucesso para %s", out_file)
        return True
    except Exception as exc:
        logger.warning("ONNX: Falha ao exportar modelo para %s: %s", out_file, exc)
        return False


def export_checkpoint_to_onnx(
    checkpoint_path: Path | str,
    output_path: Path | str | None = None,
) -> Path | None:
    """Carrega checkpoint PyTorch e converte automaticamente para ONNX."""
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        return None

    target = Path(output_path) if output_path is not None else ckpt.with_suffix(".onnx")
    loaded = load_model_checkpoint(ckpt)
    if loaded is None:
        return None

    model, _, _, _, lookback, _, _, _, _, _ = loaded
    success = export_tcn_to_onnx(model, target, lookback=lookback)
    return target if success else None
