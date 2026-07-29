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
from src.application.services.orchestrator.engine_mode import training_enabled
from src.infrastructure.inference.triton_inference_client import triton_enabled
from src.infrastructure.inference.triton_model_sync import sync_all_symbols_to_triton
from src.infrastructure.storage.torchscript_sanity import (
    verify_triton_healthcheck_async,
    verify_triton_schema_alignment_async,
    verify_triton_stressed_inference_async,
)


logger = logging.getLogger("AETH")


async def _light_infra_model_healthcheck(orch) -> None:
    """Ping leve MinIO e Triton sem probes de estresse na reconexao."""
    infra = getattr(orch, "infra", None)
    store = getattr(orch, "model_store", None)
    if infra is not None and getattr(infra, "enabled", False) and store is not None:
        head = getattr(store, "head", None)
        if callable(head):
            ok = await head()
            if not ok:
                raise ConnectionError("MINIO indisponivel")
            logger.debug("MINIO | healthcheck ok")
    if triton_enabled(orch.config):
        await verify_triton_healthcheck_async(orch.config)
        logger.debug("TRITON | healthcheck ok")


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


def _triton_ready_symbols(orch, dl_config: dict, sanity_ok: list[str]) -> bool:
    """Indica se ha TorchScript local suficiente para validar Triton."""
    if sanity_ok:
        return True
    for symbol in orch.symbols:
        ts_candidate = _scripted_path(resolve_dl_model_path(dl_config, str(symbol)))
        if ts_candidate.is_file():
            return True
    return False


async def _validate_symbol_torchscripts(
    orch,
    *,
    dl_config: dict,
    params: dict,
    arch: str,
    lookback: int,
    use_ts: bool,
) -> list[str]:
    """Baixa e valida TorchScript de cada simbolo no boot inicial."""
    store = getattr(orch, "model_store", None)
    sanity_ok: list[str] = []
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
            sanity_ok.append(sym)
        elif use_ts and not ts_path.is_file():
            logger.warning("DL: TorchScript ausente para %s; inferencia eager no runtime", sym)
    return sanity_ok


async def _bootstrap_triton_models(orch, *, lookback: int, triton_ready: bool) -> None:
    """Sincroniza e valida Triton quando habilitado."""
    if not triton_enabled(orch.config):
        return
    await sync_all_symbols_to_triton(orch)
    if triton_ready:
        probe_symbol = str(orch.symbols[0]) if orch.symbols else ""
        if not probe_symbol:
            return
        await verify_triton_schema_alignment_async(
            orch.config,
            probe_symbol,
            host_feature_dim=FEATURE_DIM,
            host_lookback=lookback,
        )
        await verify_triton_stressed_inference_async(
            orch.config,
            [str(s) for s in orch.symbols],
            lookback=lookback,
            feature_dim=FEATURE_DIM,
        )
        logger.info(
            "TRITON | schema+stress ok | %s | lb=%d fd=%d",
            probe_symbol,
            lookback,
            FEATURE_DIM,
        )
        return
    if training_enabled(orch):
        logger.info("TRITON: verificacao adiada (sem TorchScript; sessao de treino)")
        return
    raise ConnectionError("TRITON: TorchScript ausente para inferencia. Execute train.py antes de operar.")


async def bootstrap_and_validate_models(orch, *, is_initial_boot: bool | None = None) -> None:
    """Baixa checkpoints, TorchScript e valida forward pass antes do WebSocket."""
    boot = is_initial_boot if is_initial_boot is not None else bool(getattr(orch, "_is_initial_boot", True))
    if not boot:
        await _light_infra_model_healthcheck(orch)
        return
    dl_config = orch.config.get("deep_learning") or {}
    data_config = orch.config.get("data_handler") or {}
    risk_params = (orch.config.get("risk_management") or {}).get("params") or {}
    params = parse_dl_params(dl_config, data_config, risk_params)
    arch = str(params.get("arch", dl_config.get("arch", "tcn")))
    lookback = int(params.get("lookback", 48))
    use_ts = bool(params.get("use_torchscript", dl_config.get("use_torchscript", False)))
    sanity_ok = await _validate_symbol_torchscripts(
        orch,
        dl_config=dl_config,
        params=params,
        arch=arch,
        lookback=lookback,
        use_ts=use_ts,
    )
    if sanity_ok:
        logger.info("MINIO | %d TorchScript ok | %s", len(sanity_ok), ",".join(sanity_ok))
    triton_ready = _triton_ready_symbols(orch, dl_config, sanity_ok)
    await _bootstrap_triton_models(orch, lookback=lookback, triton_ready=triton_ready)


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
    """Agenda upload assincrono no event loop principal do orquestrador de forma thread-safe."""
    infra = getattr(orch, "infra", None)
    if infra is None or not infra.enabled:
        return
    main_loop = getattr(orch, "loop", None)
    if main_loop is not None and not hasattr(main_loop, "mock_calls") and main_loop.is_running():
        try:
            coro = upload_model_checkpoint(orch, symbol, local_path, arch=arch, metadata=metadata)
            main_loop.call_soon_threadsafe(lambda: main_loop.create_task(coro))
            return
        except Exception as exc:
            logger.debug("Falha ao agendar o upload do modelo no event loop principal: %s", exc)
    try:
        loop = asyncio.get_running_loop()
        coro = upload_model_checkpoint(orch, symbol, local_path, arch=arch, metadata=metadata)
        loop.create_task(coro)
    except RuntimeError:
        coro = upload_model_checkpoint(orch, symbol, local_path, arch=arch, metadata=metadata)
        asyncio.run(coro)
