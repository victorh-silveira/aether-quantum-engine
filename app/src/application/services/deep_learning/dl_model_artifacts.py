"""Download e upload de checkpoints via ModelArtifactStore."""

from __future__ import annotations

import asyncio
import logging
from inspect import iscoroutinefunction
from pathlib import Path
from typing import Any

from src.application.services.deep_learning.dl_features import FEATURE_DIM
from src.application.services.deep_learning.dl_model_checkpoint import _scripted_path
from src.application.services.deep_learning.dl_params import parse_dl_params
from src.application.services.deep_learning.dl_symbol_runtime import resolve_dl_model_path
from src.infrastructure.inference.triton_inference_client import triton_enabled
from src.infrastructure.inference.triton_model_sync import sync_all_symbols_to_triton


logger = logging.getLogger("AETH")


async def ensure_local_model_checkpoint(orch, symbol: str, dl_config: dict, params: dict) -> Path:
    """Garante checkpoint local, baixando do store quando infra habilitada."""
    path = resolve_dl_model_path(dl_config, symbol)
    infra = getattr(orch, "infra", None)
    if infra is None or not infra.enabled:
        return path
    arch = str(params.get("arch", dl_config.get("arch", "tcn")))
    ok = await orch.model_store.download_latest(symbol, arch=arch, dest=path)
    if ok:
        logger.debug("DL: checkpoint %s baixado de object storage", symbol)
    return path


async def bootstrap_and_validate_models(orch) -> None:
    """Baixa checkpoints, TorchScript e valida forward pass antes do WebSocket."""
    dl_config = orch.config.get("deep_learning") or {}
    data_config = orch.config.get("data_handler") or {}
    risk_params = (orch.config.get("risk_management") or {}).get("params") or {}
    params = parse_dl_params(dl_config, data_config, risk_params)
    arch = str(params.get("arch", dl_config.get("arch", "tcn")))
    lookback = int(params.get("lookback", 48))
    use_ts = bool(params.get("use_torchscript", dl_config.get("use_torchscript", False)))
    store = getattr(orch, "model_store", None)
    for symbol in orch.symbols:
        sym = str(symbol)
        path = await ensure_local_model_checkpoint(orch, sym, dl_config, params)
        ts_path = _scripted_path(path)
        download_ts = getattr(store, "download_torchscript", None)
        if callable(download_ts):
            await download_ts(sym, arch=arch, dest=ts_path)
        sanity = getattr(store, "sanity_check_torchscript", None)
        load_manifest = getattr(store, "load_manifest", None)
        manifest: dict[str, Any] = {}
        if callable(load_manifest) and iscoroutinefunction(load_manifest):
            manifest = await load_manifest(sym, arch=arch)
        if ts_path.is_file() and callable(sanity):
            await sanity(
                ts_path,
                lookback=lookback,
                feature_dim=FEATURE_DIM,
                symbol=sym,
                manifest=manifest or None,
            )
        elif use_ts and not ts_path.is_file():
            logger.warning("DL: TorchScript ausente para %s; inferencia eager no runtime", sym)
    if triton_enabled(orch.config):
        await sync_all_symbols_to_triton(orch)


async def upload_model_checkpoint(
    orch,
    symbol: str,
    local_path: Path,
    *,
    arch: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Envia checkpoint local para object storage quando infra habilitada."""
    infra = getattr(orch, "infra", None)
    if infra is None or not infra.enabled:
        return
    await orch.model_store.upload(symbol, local_path, arch=arch, metadata=metadata)


async def upload_all_symbol_checkpoints(orch) -> None:
    """Upload de todos os simbolos configurados apos sessao de treino."""
    dl_config = orch.config.get("deep_learning") or {}
    params_chunk = orch.config.get("data_handler") or {}
    risk_params = (orch.config.get("risk_management") or {}).get("params") or {}
    params = parse_dl_params(dl_config, params_chunk, risk_params)
    arch = str(params.get("arch", dl_config.get("arch", "tcn")))
    for symbol in orch.symbols:
        path = resolve_dl_model_path(dl_config, str(symbol))
        if not path.is_file():
            continue
        await upload_model_checkpoint(orch, str(symbol), path, arch=arch, metadata={"symbol": symbol})


def schedule_model_upload(
    orch,
    symbol: str,
    local_path: Path,
    *,
    arch: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Agenda upload assincrono quando ha event loop ativo."""
    infra = getattr(orch, "infra", None)
    if infra is None or not infra.enabled:
        return
    coro = upload_model_checkpoint(orch, symbol, local_path, arch=arch, metadata=metadata)
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(coro)
    except RuntimeError:
        asyncio.run(coro)
