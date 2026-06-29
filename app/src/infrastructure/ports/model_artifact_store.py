"""Porta de artefatos de modelo Deep Learning."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ModelArtifactStore(Protocol):
    """Contrato async para upload e download de checkpoints."""

    async def upload(
        self,
        symbol: str,
        local_path: Path,
        *,
        arch: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Envia checkpoint local para storage remoto."""

    async def download_latest(self, symbol: str, *, arch: str, dest: Path) -> bool:
        """Baixa ultimo checkpoint para destino local; False se ausente."""

    async def download_torchscript(self, symbol: str, *, arch: str, dest: Path) -> bool:
        """Baixa TorchScript para destino local; False se ausente."""

    async def sanity_check_torchscript(
        self,
        dest_ts: Path,
        *,
        lookback: int,
        feature_dim: int,
        symbol: str = "",
    ) -> None:
        """Valida artefato TorchScript com forward pass dummy."""

    async def head(self) -> bool:
        """Valida bucket ou diretorio base."""

    async def close(self) -> None:
        """Libera recursos."""
