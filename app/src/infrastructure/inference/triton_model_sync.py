"""Sincroniza artefatos TorchScript do MinIO para o repositorio Triton."""

from __future__ import annotations

import asyncio
import filecmp
import logging
import shutil
from pathlib import Path
from typing import Any

from aether_paths import repo_path
from src.application.services.deep_learning.dl_features import FEATURE_DIM
from src.application.services.deep_learning.dl_model_checkpoint import _scripted_path
from src.application.services.deep_learning.dl_params import parse_dl_params
from src.application.services.deep_learning.dl_symbol_runtime import resolve_dl_model_path
from src.infrastructure.inference.triton_inference_client import (
    triton_enabled,
    wait_triton_models_stable,
)


logger = logging.getLogger("AETH")


def default_triton_repo_path() -> Path:
    """Retorna caminho padrao do model repository montado no container."""
    return repo_path("infra", "docker", "triton-models")


def triton_config_pbtxt(*, lookback: int, feature_dim: int = FEATURE_DIM) -> str:
    """Gera config.pbtxt para backend pytorch com TorchScript LibTorch."""
    return (
        f'name: "{{name}}"\n'
        'backend: "pytorch"\n'
        'platform: "pytorch_libtorch"\n'
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


def _files_identical(left: Path, right: Path) -> bool:
    """Compara conteudo binario de dois artefatos TorchScript."""
    if not left.is_file() or not right.is_file():
        return False
    if left.stat().st_size != right.stat().st_size:
        return False
    return filecmp.cmp(left, right, shallow=False)


def _write_triton_model_dir(
    repo: Path,
    symbol: str,
    ts_source: Path,
    *,
    lookback: int,
    feature_dim: int,
) -> bool:
    """Grava model.pt e config.pbtxt sob {repo}/{symbol}/1/ quando houver drift."""
    model_dir = repo / symbol / "1"
    model_dir.mkdir(parents=True, exist_ok=True)
    dest_pt = model_dir / "model.pt"
    changed = False
    if not _files_identical(ts_source, dest_pt):
        tmp_pt = model_dir / "model.pt.tmp"
        shutil.copy2(ts_source, tmp_pt)
        tmp_pt.replace(dest_pt)
        changed = True
    pbtxt = triton_config_pbtxt(lookback=lookback, feature_dim=feature_dim).replace("{name}", symbol)
    pbtxt_path = repo / symbol / "config.pbtxt"
    existing = pbtxt_path.read_text(encoding="utf-8") if pbtxt_path.is_file() else ""
    if existing != pbtxt:
        pbtxt_path.write_text(pbtxt, encoding="utf-8")
        changed = True
    return changed


async def sync_symbol_torchscript_to_triton(
    store: Any,
    symbol: str,
    *,
    arch: str,
    local_ts_path: Path,
    lookback: int,
    feature_dim: int = FEATURE_DIM,
    repo_path_override: Path | None = None,
) -> tuple[bool, bool]:
    """Copia latest_ts.pt local para o layout de modelos do Triton."""
    repo = repo_path_override or default_triton_repo_path()
    if not local_ts_path.is_file():
        download_ts = getattr(store, "download_torchscript", None)
        if callable(download_ts):
            ok = await download_ts(symbol, arch=arch, dest=local_ts_path)
            if not ok:
                logger.warning("TRITON: TorchScript ausente para %s", symbol)
                return False, False
        else:
            return False, False

    def _sync() -> bool:
        """Grava artefatos no layout Triton em thread de I/O."""
        return _write_triton_model_dir(repo, symbol, local_ts_path, lookback=lookback, feature_dim=feature_dim)

    repo_changed = await asyncio.to_thread(_sync)
    if repo_changed:
        logger.debug("TRITON: modelo %s sincronizado em %s", symbol, repo / symbol)
    else:
        logger.debug("TRITON: modelo %s inalterado em %s", symbol, repo / symbol)
    return True, repo_changed


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
    synced_symbols: list[str] = []
    repo_changed = False
    for symbol in orch.symbols:
        sym = str(symbol)
        path = resolve_dl_model_path(dl_config, sym)
        ts_path = _scripted_path(path)
        ok, symbol_changed = await sync_symbol_torchscript_to_triton(
            store,
            sym,
            arch=arch,
            local_ts_path=ts_path,
            lookback=lookback,
            repo_path_override=repo,
        )
        if ok:
            synced_symbols.append(sym)
            repo_changed = repo_changed or symbol_changed
    if not synced_symbols:
        return
    label = ",".join(synced_symbols)
    if triton_enabled(orch.config):
        repo_ok = await wait_triton_models_stable(orch.config, synced_symbols, repo_changed=repo_changed)
        status = "ready" if repo_ok else "timeout"
        drift = "sync" if repo_changed else "cache"
        logger.info("TRITON | %d modelos | %s | %s | %s", len(synced_symbols), label, status, drift)
        if not repo_ok:
            raise ConnectionError(f"TRITON: modelos nao ficaram prontos: {label}")
    else:
        logger.info("TRITON | %d modelos | %s", len(synced_symbols), label)
