"""Armazenamento local de checkpoints para testes e cache."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from aether_paths import repo_path


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

    async def head(self) -> bool:
        """Verifica se diretorio base existe."""
        return self._base.is_dir()

    async def close(self) -> None:
        """Encerramento sem efeito."""
        return
