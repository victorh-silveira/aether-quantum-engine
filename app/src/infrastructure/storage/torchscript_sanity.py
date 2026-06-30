"""Forward pass de validacao para artefatos TorchScript e inferencia Triton."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.infrastructure.inference.triton_grpc_client import get_triton_grpc_client
from src.infrastructure.inference.triton_inference_client import triton_grpc_url
from src.infrastructure.inference.triton_model_metadata import (
    fetch_triton_model_metadata_async,
    parse_triton_input_dims,
)
from src.infrastructure.storage.torchscript_sanity_probes import (
    build_sanity_probe_tensors,
    build_stressed_regime_probe_ndarray,
)


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


def assert_triton_host_schema_aligned(
    model_payload: dict[str, Any],
    *,
    host_feature_dim: int,
    host_lookback: int,
    model_name: str,
) -> None:
    """Falha se dims do Triton divergirem do schema estatico do host."""
    triton_feature_dim, triton_lookback = parse_triton_input_dims(model_payload)
    if triton_feature_dim != int(host_feature_dim):
        raise RuntimeError(
            "TRITON schema drift: modelo="
            f"{model_name} feature_dim={triton_feature_dim} "
            f"diverge do host FEATURE_DIM={host_feature_dim}"
        )
    if triton_lookback is not None and triton_lookback != int(host_lookback):
        raise RuntimeError(
            "TRITON schema drift: modelo="
            f"{model_name} lookback={triton_lookback} "
            f"diverge do host lookback={host_lookback}"
        )


async def verify_triton_schema_alignment_async(
    config: dict,
    model_name: str,
    *,
    host_feature_dim: int,
    host_lookback: int,
) -> None:
    """Consulta Triton HTTP e valida alinhamento de payload com o host."""
    payload = await fetch_triton_model_metadata_async(config, str(model_name))
    assert_triton_host_schema_aligned(
        payload,
        host_feature_dim=host_feature_dim,
        host_lookback=host_lookback,
        model_name=str(model_name),
    )


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


def assert_triton_probability(value: float, *, model_name: str) -> float:
    """Falha se probabilidade Triton estiver fora de [0,1] ou nao for finita."""
    prob = float(value)
    if not np.isfinite(prob):
        raise RuntimeError(f"Triton model={model_name} produziu NaN ou Inf")
    if prob < 0.0 or prob > 1.0:
        raise RuntimeError(f"Triton model={model_name} probabilidade={prob} fora de [0,1]")
    return prob


async def verify_triton_stressed_inference_async(
    config: dict,
    symbols: list[str],
    *,
    lookback: int,
    feature_dim: int,
) -> None:
    """Fail-fast se Triton retornar NaN/Inf ou probabilidade fora de [0,1] sob regime estressado."""
    if not symbols:
        return
    tensor = build_stressed_regime_probe_ndarray(lookback, feature_dim)
    client = await get_triton_grpc_client(triton_grpc_url(config))
    batch = {str(sym): tensor for sym in symbols}
    probs = await client.infer_symbols_concurrent(batch)
    for sym in symbols:
        raw = probs.get(str(sym))
        if raw is None:
            raise RuntimeError(f"Triton model={sym} sem resposta no sanity estressado")
        assert_triton_probability(float(raw), model_name=str(sym))


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
