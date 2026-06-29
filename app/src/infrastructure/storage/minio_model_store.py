"""Armazenamento MinIO para checkpoints Deep Learning."""

from __future__ import annotations

import asyncio
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Any

from minio import Minio


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

        def _do_upload() -> None:
            """Executa upload sincrono na thread pool."""
            if not self._client.bucket_exists(self._bucket):
                self._client.make_bucket(self._bucket)
            self._client.fput_object(self._bucket, key, str(local_path))
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
            """Executa download sincrono na thread pool."""
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                self._client.fget_object(self._bucket, key, str(dest))
                return True
            except Exception:
                return False

        return await asyncio.to_thread(_do_download)

    async def head(self) -> bool:
        """Valida conectividade com MinIO e garante bucket configurado."""

        def _check() -> bool:
            """Verifica bucket e cria quando ausente no thread pool."""
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
