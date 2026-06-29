"""Armazenamento MinIO para checkpoints Deep Learning."""

from __future__ import annotations

import asyncio
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Any

from minio import Minio

from src.infrastructure.storage.torchscript_sanity import verify_torchscript_artifact


class MinioModelStore:
    """Upload e download de checkpoints via SDK MinIO."""

    def __init__(
        self,
        *,
        endpoint: str,
        bucket: str,
        access_key: str,
        secret_key: str,
        secure: bool = False,
    ):
        host = str(endpoint).replace("http://", "").replace("https://", "")
        self._bucket = str(bucket)
        self._client = Minio(host, access_key=access_key, secret_key=secret_key, secure=bool(secure))
        self.logger = logging.getLogger("AETH")

    def _object_key(self, symbol: str, arch: str, name: str) -> str:
        """Monta chave de objeto no bucket MinIO."""
        return f"{symbol}/{arch}/{name}"

    async def upload(
        self,
        symbol: str,
        local_path: Path,
        *,
        arch: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Envia checkpoint e manifest para MinIO."""
        key = self._object_key(symbol, arch, "latest.pth")
        manifest_key = f"{symbol}/manifest.json"
        ts_path = local_path.with_name(f"{local_path.stem}_ts.pt")

        def _do_upload() -> None:
            """Executa upload sincrono no thread pool."""
            if not self._client.bucket_exists(self._bucket):
                self._client.make_bucket(self._bucket)
            self._client.fput_object(self._bucket, key, str(local_path))
            if ts_path.is_file():
                ts_key = self._object_key(symbol, arch, "latest_ts.pt")
                self._client.fput_object(self._bucket, ts_key, str(ts_path))
            payload = dict(metadata or {})
            data = json.dumps(payload).encode("utf-8")
            self._client.put_object(
                self._bucket,
                manifest_key,
                BytesIO(data),
                len(data),
                content_type="application/json",
            )

        await asyncio.to_thread(_do_upload)

    async def download_latest(self, symbol: str, *, arch: str, dest: Path) -> bool:
        """Baixa latest.pth do MinIO para cache local."""
        key = self._object_key(symbol, arch, "latest.pth")

        def _do_download() -> bool:
            """Baixa latest.pth para destino local."""
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                self._client.fget_object(self._bucket, key, str(dest))
                return True
            except Exception:
                return False

        return await asyncio.to_thread(_do_download)

    async def download_torchscript(self, symbol: str, *, arch: str, dest: Path) -> bool:
        """Baixa latest_ts.pt do MinIO para cache local."""
        key = self._object_key(symbol, arch, "latest_ts.pt")

        def _do_download() -> bool:
            """Baixa latest_ts.pt para destino local."""
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                self._client.fget_object(self._bucket, key, str(dest))
                return True
            except Exception:
                return False

        return await asyncio.to_thread(_do_download)

    async def sanity_check_torchscript(
        self,
        dest_ts: Path,
        *,
        lookback: int,
        feature_dim: int,
        symbol: str = "",
    ) -> None:
        """Executa forward pass de sanidade no artefato TorchScript."""

        def _run() -> None:
            """Executa forward pass de sanidade no thread pool."""
            verify_torchscript_artifact(dest_ts, lookback=lookback, feature_dim=feature_dim)

        await asyncio.to_thread(_run)
        label = symbol or dest_ts.stem
        self.logger.info("MINIO: sanity ok %s", label)

    async def head(self) -> bool:
        """Valida conectividade com MinIO e garante bucket configurado."""

        def _check() -> bool:
            """Verifica bucket MinIO no thread pool."""
            try:
                if not self._client.bucket_exists(self._bucket):
                    self._client.make_bucket(self._bucket)
                return True
            except Exception as exc:
                self.logger.debug("MINIO: head falhou: %s", exc)
                return False

        return await asyncio.to_thread(_check)

    async def close(self) -> None:
        """Encerramento sem efeito."""
        return
