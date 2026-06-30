"""Cliente gRPC aio com canal persistente e inferencia concorrente no Triton."""

from __future__ import annotations

import asyncio
import logging
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


logger = logging.getLogger("AETH")

_INPUT_NAME = "INPUT__0"
_OUTPUT_NAME = "OUTPUT__0"
_MAX_MSG = 512 * 1024 * 1024

_pool_lock = asyncio.Lock()


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


class TritonGrpcClient:
    """Canal gRPC aio persistente com inferencias paralelas via asyncio.gather."""

    def __init__(self) -> None:
        self._channel: grpc.aio.Channel | None = None
        self._infer: Any | None = None
        self._url: str | None = None
        self._lock = asyncio.Lock()

    @property
    def channel(self) -> grpc.aio.Channel | None:
        """Canal gRPC aio persistente quando conectado."""
        return self._channel

    async def connect(self, url: str) -> None:
        """Abre canal grpc.aio.insecure_channel persistente para o endpoint."""
        if grpc_aio is None or service_pb2_grpc is None:
            raise RuntimeError("tritonclient[grpc] nao instalado")
        target = str(url).strip()
        async with self._lock:
            if self._channel is not None and self._url == target:
                return
            await self._close_unlocked()
            self._channel = grpc.aio.insecure_channel(target, options=_channel_options())
            self._infer = _attach_channel(
                grpc_aio.InferenceServerClient.__new__(grpc_aio.InferenceServerClient), self._channel
            )
            self._url = target

    async def close(self) -> None:
        """Encerra canal e stub gRPC aio."""
        async with self._lock:
            await self._close_unlocked()

    async def _close_unlocked(self) -> None:
        """Fecha recursos sem adquirir lock interno."""
        if self._infer is not None:
            await self._infer.close()
        elif self._channel is not None:
            await self._channel.close()
        self._infer = None
        self._channel = None
        self._url = None

    async def infer_symbol(self, model_name: str, tensor: np.ndarray) -> float:
        """Executa inferencia gRPC para um simbolo."""
        if self._infer is None:
            raise RuntimeError("TritonGrpcClient nao conectado")
        batch = np.asarray(tensor, dtype=np.float32)
        if batch.ndim == 2:
            batch = batch.reshape(1, batch.shape[0], batch.shape[1])
        inputs = [grpc_aio.InferInput(_INPUT_NAME, batch.shape, "FP32")]
        inputs[0].set_data_from_numpy(batch)
        outputs = [grpc_aio.InferRequestedOutput(_OUTPUT_NAME)]
        try:
            result = await self._infer.infer(model_name=str(model_name), inputs=inputs, outputs=outputs)
            return _parse_raw_output(result)
        except InferenceServerException as exc:
            logger.error("TRITON: inferencia falhou para %s: %s", model_name, exc)
            raise

    async def infer_symbols_concurrent(self, tensors: dict[str, np.ndarray]) -> dict[str, float]:
        """Dispara inferencias em paralelo via asyncio.gather."""

        async def _one(sym: str, arr: np.ndarray) -> tuple[str, float]:
            """Executa inferencia para um par simbolo/tensor."""
            prob = await self.infer_symbol(sym, arr)
            return sym, prob

        if not tensors:
            return {}
        pairs = await asyncio.gather(*[_one(sym, arr) for sym, arr in tensors.items()])
        return dict(pairs)


async def get_triton_grpc_client(url: str) -> TritonGrpcClient:
    """Retorna cliente gRPC aio singleton para o endpoint."""
    async with _pool_lock:
        if _GrpcClientPool.client is None:
            _GrpcClientPool.client = TritonGrpcClient()
        await _GrpcClientPool.client.connect(str(url))
        return _GrpcClientPool.client


async def close_triton_grpc_client() -> None:
    """Fecha cliente gRPC aio singleton se aberto."""
    async with _pool_lock:
        if _GrpcClientPool.client is not None:
            await _GrpcClientPool.client.close()
            _GrpcClientPool.client = None
