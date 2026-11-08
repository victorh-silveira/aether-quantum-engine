"""Forward pass de validacao para artefatos TorchScript locais."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.infrastructure.storage.torchscript_sanity_probes import build_sanity_probe_tensors


def validate_manifest_schema(
    manifest: dict[str, Any] | None,
    *,
    lookback: int,
    feature_dim: int,
) -> None:
    """Falha se metadados do model store divergirem do schema esperado."""
    if not isinstance(manifest, dict) or not manifest:
        return
    declared = manifest.get("feature_dim", manifest.get("input_dim"))
    if declared is not None and int(declared) != int(feature_dim):
        raise RuntimeError(f"Manifest feature_dim={declared} incompativel com esperado={feature_dim}")
    manifest_lookback = manifest.get("lookback")
    if manifest_lookback is not None and int(manifest_lookback) != int(lookback):
        raise RuntimeError(f"Manifest lookback={manifest_lookback} incompativel com esperado={lookback}")
    for key in ("norm_mean", "norm_std"):
        values = manifest.get(key)
        if values is not None and len(values) != int(feature_dim):
            raise RuntimeError(f"Manifest {key} length={len(values)} != feature_dim={feature_dim}")


def _tensor_values(out: Any) -> np.ndarray:
    """Converte saida do modelo em vetor numpy unidimensional."""
    if isinstance(out, (tuple, list)):
        out = out[0]
    if not torch.is_tensor(out):
        raise RuntimeError(f"TorchScript retornou tipo invalido: {type(out).__name__}")
    return np.asarray(out.detach().cpu().numpy(), dtype=np.float64).reshape(-1)


def _assert_forward_output(out: Any, *, probe: str) -> None:
    """Valida tensor de saida do forward pass para um probe nomeado."""
    flat = _tensor_values(out)
    if flat.size < 1:
        raise RuntimeError(f"TorchScript probe={probe} shape invalida")
    if not np.isfinite(flat).all():
        raise RuntimeError(f"TorchScript probe={probe} produziu NaN ou Inf")


def _assert_stressed_probability(out: Any, *, probe: str) -> None:
    """Valida saida sob regime estressado sem NaN ou Inf."""
    _assert_forward_output(out, probe=probe)


async def verify_torchscript_artifact_light_async(
    path: Path,
    *,
    lookback: int,
    feature_dim: int,
    manifest: dict[str, Any] | None = None,
) -> None:
    """Valida presenca do artefato e schema do manifest sem probes de estresse."""
    if not path.is_file():
        raise RuntimeError(f"TorchScript ausente: {path}")
    validate_manifest_schema(manifest, lookback=lookback, feature_dim=feature_dim)


def verify_torchscript_artifact(
    path: Path,
    *,
    lookback: int,
    feature_dim: int,
    manifest: dict[str, Any] | None = None,
) -> None:
    """Falha se TorchScript nao aceitar probes ou retornar NaN/Inf."""
    if not path.is_file():
        raise RuntimeError(f"TorchScript ausente: {path}")
    validate_manifest_schema(manifest, lookback=lookback, feature_dim=feature_dim)
    model = torch.jit.load(str(path), map_location=torch.device("cpu"))
    model.eval()
    probes = build_sanity_probe_tensors(lookback, feature_dim)
    with torch.no_grad():
        for probe_name, tensor in probes:
            out = model(tensor)
            if probe_name == "stressed_regime":
                _assert_stressed_probability(out, probe=probe_name)
            else:
                _assert_forward_output(out, probe=probe_name)


async def verify_torchscript_artifact_async(
    path: Path,
    *,
    lookback: int,
    feature_dim: int,
    manifest: dict[str, Any] | None = None,
) -> None:
    """Executa verify_torchscript_artifact no thread pool."""
    await asyncio.to_thread(
        verify_torchscript_artifact,
        path,
        lookback=lookback,
        feature_dim=feature_dim,
        manifest=manifest,
    )
