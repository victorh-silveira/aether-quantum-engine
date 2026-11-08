"""Armazenamento local de checkpoints para testes e cache."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from aether_paths import repo_path
from src.infrastructure.storage.torchscript_sanity import verify_torchscript_artifact_async


logger = logging.getLogger("AETH")


class LocalModelStore:
    """Copia checkpoints no diretorio data/dl sem object storage."""

    def __init__(self, base_dir: str | Path | None = None):
        self._base = Path(base_dir) if base_dir is not None else repo_path("data", "dl")
        self._base.mkdir(parents=True, exist_ok=True)

    def _object_dir(self, symbol: str, arch: str) -> Path:
        """Retorna diretorio local do simbolo e arquitetura."""
        return self._base / str(symbol) / str(arch)

    async def upload(
        self,
        symbol: str,
        local_path: Path,
        *,
        arch: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Copia checkpoint e manifest para cache local."""
        dest_dir = self._object_dir(symbol, arch)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / "latest.pth"
        shutil.copy2(local_path, dest)
        manifest = dest_dir / "manifest.json"
        payload = dict(metadata or {})
        manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    async def download_latest(self, symbol: str, *, arch: str, dest: Path) -> bool:
        """Copia latest.pth local para destino de inferencia."""
        src = self._object_dir(symbol, arch) / "latest.pth"
        if not src.is_file():
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return True

    async def download_torchscript(self, symbol: str, *, arch: str, dest: Path) -> bool:
        """Copia latest_ts.pt local para destino de inferencia."""
        src = self._object_dir(symbol, arch) / "latest_ts.pt"
        if not src.is_file():
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return True

    async def load_manifest(self, symbol: str, *, arch: str) -> dict[str, Any]:
        """Carrega manifest.json local do simbolo."""
        manifest = self._object_dir(symbol, arch) / "manifest.json"
        if not manifest.is_file():
            return {}
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return {}

    async def sanity_check_torchscript(
        self,
        dest_ts: Path,
        *,
        lookback: int,
        feature_dim: int,
        symbol: str = "",
        manifest: dict[str, Any] | None = None,
    ) -> None:
        """Valida forward pass do TorchScript local."""
        await verify_torchscript_artifact_async(
            dest_ts,
            lookback=lookback,
            feature_dim=feature_dim,
            manifest=manifest,
        )
        label = symbol or dest_ts.stem
        logger.debug("LOCAL: sanity ok %s", label)

    async def head(self) -> bool:
        """Verifica se diretorio base existe."""
        return self._base.is_dir()

    async def close(self) -> None:
        """Encerramento sem efeito."""
        return
