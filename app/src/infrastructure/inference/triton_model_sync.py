"""Sincroniza artefatos TorchScript do MinIO para o repositorio Triton."""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Any

from aether_paths import repo_path
from src.application.services.deep_learning.dl_features import FEATURE_DIM
from src.application.services.deep_learning.dl_model_checkpoint import _scripted_path
from src.application.services.deep_learning.dl_params import parse_dl_params
from src.application.services.deep_learning.dl_symbol_runtime import resolve_dl_model_path


logger = logging.getLogger("AETH")


def default_triton_repo_path() -> Path:
    """Retorna caminho padrao do model repository montado no container."""
    return repo_path("infra", "docker", "triton-models")


def triton_config_pbtxt(*, lookback: int, feature_dim: int = FEATURE_DIM) -> str:
    """Gera config.pbtxt para backend pytorch_libtorch."""
    return (
        f'name: "{{name}}"\n'
        'backend: "pytorch_libtorch"\n'
        "max_batch_size: 8\n"
        "input [\n"
        "  {\n"
        '    name: "INPUT__0"\n'
        "    data_type: TYPE_FP32\n"
        f"    dims: [ {lookback}, {feature_dim} ]\n"
        "  }\n"
        "]\n"
        "output [\n"
        "  {\n"
        '    name: "OUTPUT__0"\n'
        "    data_type: TYPE_FP32\n"
        "    dims: [ 1 ]\n"
        "  }\n"
        "]\n"
    )


def _write_triton_model_dir(
    repo: Path,
    symbol: str,
    ts_source: Path,
    *,
    lookback: int,
    feature_dim: int,
) -> None:
    """Grava model.pt e config.pbtxt sob {repo}/{symbol}/1/."""
    model_dir = repo / symbol / "1"
    model_dir.mkdir(parents=True, exist_ok=True)
    dest_pt = model_dir / "model.pt"
    shutil.copy2(ts_source, dest_pt)
    pbtxt = triton_config_pbtxt(lookback=lookback, feature_dim=feature_dim).replace("{name}", symbol)
    (repo / symbol / "config.pbtxt").write_text(pbtxt, encoding="utf-8")


async def sync_symbol_torchscript_to_triton(
    store: Any,
    symbol: str,
    *,
    arch: str,
    local_ts_path: Path,
    lookback: int,
    feature_dim: int = FEATURE_DIM,
    repo_path_override: Path | None = None,
) -> bool:
    """Copia latest_ts.pt local para o layout de modelos do Triton."""
    repo = repo_path_override or default_triton_repo_path()
    if not local_ts_path.is_file():
        download_ts = getattr(store, "download_torchscript", None)
        if callable(download_ts):
            ok = await download_ts(symbol, arch=arch, dest=local_ts_path)
            if not ok:
                logger.warning("TRITON: TorchScript ausente para %s", symbol)
                return False
        else:
            return False

    def _sync() -> None:
        """Grava artefatos no layout Triton em thread de I/O."""
        _write_triton_model_dir(repo, symbol, local_ts_path, lookback=lookback, feature_dim=feature_dim)

    await asyncio.to_thread(_sync)
    logger.debug("TRITON: modelo %s sincronizado em %s", symbol, repo / symbol)
    return True


async def sync_all_symbols_to_triton(
    orch: Any,
    *,
    repo_path_override: Path | None = None,
) -> None:
    """Sincroniza todos os simbolos configurados para o repositorio Triton."""
    dl_config = orch.config.get("deep_learning") or {}
    data_config = orch.config.get("data_handler") or {}
    risk_params = (orch.config.get("risk_management") or {}).get("params") or {}
    params = parse_dl_params(dl_config, data_config, risk_params)
    arch = str(params.get("arch", dl_config.get("arch", "tcn")))
    lookback = int(params.get("lookback", 48))
    store = getattr(orch, "model_store", None)
    if store is None:
        return
    repo = repo_path_override or default_triton_repo_path()
    for symbol in orch.symbols:
        sym = str(symbol)
        path = resolve_dl_model_path(dl_config, sym)
        ts_path = _scripted_path(path)
        await sync_symbol_torchscript_to_triton(
            store,
            sym,
            arch=arch,
            local_ts_path=ts_path,
            lookback=lookback,
            repo_path_override=repo,
        )
