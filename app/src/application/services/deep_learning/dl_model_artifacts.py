"""Download e upload de checkpoints via ModelArtifactStore."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from src.application.services.deep_learning.dl_params import parse_dl_params
from src.application.services.deep_learning.dl_symbol_runtime import resolve_dl_model_path


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
