"""Sincroniza artefatos TorchScript do MinIO para o repositorio Triton."""

from __future__ import annotations

import asyncio
import filecmp
import logging
import os
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
_TRITON_REPO_SYNC_LOCK = asyncio.Lock()


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


def _fsync_path(path: Path) -> None:
    """Forca flush do descritor de arquivo para o bind mount do host."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        return
    finally:
        os.close(fd)


def _fsync_directory(path: Path) -> None:
    """Tenta fsync do diretorio pai (POSIX); ignora falha em FS Windows/WSL."""
    try:
        dir_fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        return
    finally:
        os.close(dir_fd)


def _atomic_publish_torchscript(src: Path, dest: Path) -> None:
    """Copia TorchScript com flush completo antes do rename atomico."""
    tmp_pt = dest.with_name(f"{dest.name}.tmp")
    with src.open("rb") as fsrc, tmp_pt.open("wb") as fdst:
        shutil.copyfileobj(fsrc, fdst, length=1024 * 1024)
        fdst.flush()
        os.fsync(fdst.fileno())
    tmp_pt.replace(dest)
    _fsync_path(dest)
    _fsync_directory(dest.parent)


def _write_text_durable(path: Path, content: str) -> None:
    """Grava texto com fsync e libera o descritor antes de qualquer /load Triton."""
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)
    _fsync_path(path)
    _fsync_directory(path.parent)


def _repo_durability_barrier(repo: Path, symbols: list[str]) -> None:
    """Garante buffers em disco e descritores liberados antes do load-over-load."""
    for symbol in symbols:
        model_pt = repo / symbol / "1" / "model.pt"
        pbtxt = repo / symbol / "config.pbtxt"
        if model_pt.is_file():
            _fsync_path(model_pt)
            _fsync_directory(model_pt.parent)
        if pbtxt.is_file():
            _fsync_path(pbtxt)
            _fsync_directory(pbtxt.parent)
    if repo.is_dir():
        _fsync_directory(repo)


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
        _atomic_publish_torchscript(ts_source, dest_pt)
        changed = True
    pbtxt = triton_config_pbtxt(lookback=lookback, feature_dim=feature_dim).replace("{name}", symbol)
    pbtxt_path = repo / symbol / "config.pbtxt"
    existing = pbtxt_path.read_text(encoding="utf-8") if pbtxt_path.is_file() else ""
    if existing != pbtxt:
        _write_text_durable(pbtxt_path, pbtxt)
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
    """Sincroniza artefatos com barreira de disco e load-over-load serializado."""
    async with _TRITON_REPO_SYNC_LOCK:
        await _sync_all_symbols_to_triton_locked(orch, repo_path_override=repo_path_override)


async def _sync_all_symbols_to_triton_locked(
    orch: Any,
    *,
    repo_path_override: Path | None = None,
) -> None:
    """Corpo do sync Triton sob lock: publica bytes, fsync, so entao /load."""
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
    changed_symbols: list[str] = []
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
            if symbol_changed:
                changed_symbols.append(sym)
    if not synced_symbols:
        return
    await asyncio.to_thread(_repo_durability_barrier, repo, synced_symbols)
    label = ",".join(synced_symbols)
    repo_changed = bool(changed_symbols)
    if triton_enabled(orch.config):
        repo_ok = await wait_triton_models_stable(
            orch.config,
            synced_symbols,
            repo_changed=repo_changed,
            changed_models=changed_symbols,
        )
        status = "ready" if repo_ok else "timeout"
        drift = "sync" if repo_changed else "cache"
        logger.info(
            "TRITON | %d modelos | %s | %s | %s | load-over-load",
            len(synced_symbols),
            label,
            status,
            drift,
        )
        if not repo_ok:
            raise ConnectionError(f"TRITON: modelos nao ficaram prontos: {label}")
    else:
        logger.info("TRITON | %d modelos | %s", len(synced_symbols), label)
