"""Cliente gRPC aio com canal persistente e inferencia concorrente no Triton."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

import grpc.aio
import numpy as np


try:
    import tritonclient.grpc.aio as grpc_aio
    from tritonclient.grpc import service_pb2_grpc
    from tritonclient.grpc._client import InferenceServerClientBase, KeepAliveOptions
    from tritonclient.utils import InferenceServerException
except ImportError:
    grpc_aio = None
    service_pb2_grpc = None
    InferenceServerClientBase = object
    KeepAliveOptions = None
    InferenceServerException = Exception


from src.application.services.infra_timing_config import resolve_triton_infer_timeout


logger = logging.getLogger("AETH")
_INPUT_NAME = "INPUT__0"
_OUTPUT_NAME = "OUTPUT__0"
_MAX_MSG = 512 * 1024 * 1024
_pool_guard = threading.Lock()


class TritonInferenceTimeout(TimeoutError):
    """Inferencia gRPC Triton excedeu o timeout configurado."""


class _GrpcClientPool:
    """Pool singleton de cliente gRPC aio."""

    client: TritonGrpcClient | None = None


def _channel_options() -> list[tuple[str, int | bool]]:
    """Opcoes de canal gRPC aio com keepalive para conexao persistente."""
    keepalive = KeepAliveOptions() if KeepAliveOptions is not None else None
    if keepalive is None:
        return [
            ("grpc.max_send_message_length", _MAX_MSG),
            ("grpc.max_receive_message_length", _MAX_MSG),
        ]
    return [
        ("grpc.max_send_message_length", _MAX_MSG),
        ("grpc.max_receive_message_length", _MAX_MSG),
        ("grpc.keepalive_time_ms", keepalive.keepalive_time_ms),
        ("grpc.keepalive_timeout_ms", keepalive.keepalive_timeout_ms),
        ("grpc.keepalive_permit_without_calls", keepalive.keepalive_permit_without_calls),
        ("grpc.http2.max_pings_without_data", keepalive.http2_max_pings_without_data),
    ]


def _attach_channel(client: Any, channel: grpc.aio.Channel) -> Any:
    """Associa stub Triton a um canal gRPC aio ja aberto."""
    InferenceServerClientBase.__init__(client)
    client._channel = channel
    client._client_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)
    client._verbose = False
    return client


def _pack_inference_tensor(tensor: np.ndarray) -> np.ndarray:
    """Normaliza shape e garante buffer FP32 contiguo para envio gRPC."""
    batch = np.asarray(tensor, dtype=np.float32)
    if batch.ndim == 2:
        batch = batch.reshape(1, batch.shape[0], batch.shape[1])
    return np.ascontiguousarray(batch)


def _parse_raw_output(result: Any) -> float:
    """Extrai probabilidade bruta finita do tensor de saida Triton."""
    out = result.as_numpy(_OUTPUT_NAME)
    if out is None:
        return 0.5
    flat = np.asarray(out, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return 0.5
    val = float(flat[0])
    if not np.isfinite(val):
        return 0.5
    return max(0.0, min(1.0, val))


def _running_loop_or_none() -> asyncio.AbstractEventLoop | None:
    """Retorna o event loop em execucao ou None fora de contexto async."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


class TritonGrpcClient:
    """Canal gRPC aio persistente com inferencias paralelas via asyncio.gather."""

    def __init__(self) -> None:
        self._channel: grpc.aio.Channel | None = None
        self._infer: Any | None = None
        self._url: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock: asyncio.Lock | None = None

    @property
    def channel(self) -> grpc.aio.Channel | None:
        """Canal gRPC aio persistente quando conectado."""
        return self._channel

    def bound_to_running_loop(self) -> bool:
        """True quando o canal esta vinculado ao event loop corrente."""
        running = _running_loop_or_none()
        return running is not None and self._loop is running and self._channel is not None

    def abandon(self) -> None:
        """Descarta recursos sem await quando o loop original ja encerrou."""
        self._infer = None
        self._channel = None
        self._url = None
        self._loop = None
        self._lock = None

    def _loop_lock(self) -> asyncio.Lock:
        """Lock asyncio vinculado ao event loop corrente."""
        loop = asyncio.get_running_loop()
        if self._lock is None or self._loop is not loop:
            self._lock = asyncio.Lock()
        return self._lock

    async def connect(self, url: str) -> None:
        """Abre canal grpc.aio.insecure_channel persistente para o endpoint."""
        if grpc_aio is None or service_pb2_grpc is None:
            raise RuntimeError("tritonclient[grpc] nao instalado")
        target = str(url).strip()
        loop = asyncio.get_running_loop()
        async with self._loop_lock():
            if self._channel is not None and self._url == target and self._loop is loop:
                return
            await self._close_unlocked()
            self._channel = grpc.aio.insecure_channel(target, options=_channel_options())
            self._infer = _attach_channel(
                grpc_aio.InferenceServerClient.__new__(grpc_aio.InferenceServerClient),
                self._channel,
            )
            self._url = target
            self._loop = loop

    async def close(self) -> None:
        """Encerra canal e stub gRPC aio."""
        running = _running_loop_or_none()
        if self._loop is not None and running is not None and self._loop is not running:
            self.abandon()
            return
        async with self._loop_lock():
            await self._close_unlocked()

    async def _close_unlocked(self) -> None:
        """Fecha recursos sem adquirir lock interno."""
        running = _running_loop_or_none()
        if self._loop is not None and running is not None and self._loop is not running:
            self.abandon()
            return
        try:
            if self._infer is not None:
                await self._infer.close()
            elif self._channel is not None:
                await self._channel.close()
        except RuntimeError:
            self.abandon()
            return
        self.abandon()

    async def close_channel(self) -> None:
        """Encerra canal gRPC aio persistente."""
        await self.close()

    @classmethod
    async def close_channel_pool(cls) -> None:
        """Fecha pool singleton gRPC aio."""
        await close_triton_grpc_client()

    async def infer_symbol(
        self,
        model_name: str,
        tensor: np.ndarray,
        *,
        timeout: float = resolve_triton_infer_timeout(),
    ) -> float:
        """Executa inferencia gRPC para um simbolo com timeout rigido."""
        if self._infer is None:
            raise RuntimeError("TritonGrpcClient nao conectado")
        batch = _pack_inference_tensor(tensor)
        inputs = [grpc_aio.InferInput(_INPUT_NAME, batch.shape, "FP32")]
        inputs[0].set_data_from_numpy(batch)
        outputs = [grpc_aio.InferRequestedOutput(_OUTPUT_NAME)]
        try:
            result = await asyncio.wait_for(
                self._infer.infer(model_name=str(model_name), inputs=inputs, outputs=outputs),
                timeout=float(timeout),
            )
            return _parse_raw_output(result)
        except TimeoutError as exc:
            raise TritonInferenceTimeout(f"Triton infer timeout {float(timeout):.3f}s for {model_name}") from exc
        except InferenceServerException as exc:
            logger.error("TRITON: inferencia falhou para %s: %s", model_name, exc)
            raise

    async def infer_symbols_concurrent(
        self,
        tensors: dict[str, np.ndarray],
        *,
        timeout: float = resolve_triton_infer_timeout(),
    ) -> dict[str, float]:
        """Dispara inferencias em paralelo com deadline unico para o lote."""

        async def _one(sym: str, arr: np.ndarray) -> tuple[str, float]:
            """Executa inferencia para um par simbolo/tensor."""
            prob = await self.infer_symbol(sym, arr, timeout=timeout)
            return sym, prob

        if not tensors:
            return {}
        try:
            pairs = await asyncio.wait_for(
                asyncio.gather(*[_one(sym, arr) for sym, arr in tensors.items()]),
                timeout=float(timeout),
            )
        except TimeoutError as exc:
            raise TritonInferenceTimeout(
                f"Triton batch infer timeout {float(timeout):.3f}s for {len(tensors)} symbols"
            ) from exc
        return dict(pairs)


async def get_triton_grpc_client(url: str) -> TritonGrpcClient:
    """Retorna cliente gRPC aio singleton recriado se o event loop mudou."""
    loop = asyncio.get_running_loop()
    target = str(url).strip()
    stale: TritonGrpcClient | None = None
    with _pool_guard:
        client = _GrpcClientPool.client
        if client is not None:
            loop_stale = (client._loop is not None and client._loop is not loop) or (
                client._channel is not None and client._loop is not loop
            )
            url_mismatch = client._url is not None and client._url != target
            if loop_stale or url_mismatch:
                stale = client
                _GrpcClientPool.client = None
                client = None
        if client is None:
            client = TritonGrpcClient()
            _GrpcClientPool.client = client
    if stale is not None:
        stale.abandon()
    await client.connect(target)
    return client


async def close_triton_grpc_client() -> None:
    """Fecha cliente gRPC aio singleton se aberto."""
    with _pool_guard:
        client = _GrpcClientPool.client
        _GrpcClientPool.client = None
    if client is not None:
        await client.close()
