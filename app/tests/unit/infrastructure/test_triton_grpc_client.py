import asyncio
import asyncio.tasks as asyncio_tasks
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.infrastructure.inference import triton_grpc_client as triton_grpc_module
from src.infrastructure.inference.triton_grpc_client import (
    _INFER_TIMEOUT_SEC,
    _MAX_MSG,
    InferenceServerException,
    TritonGrpcClient,
    _attach_channel,
    _channel_options,
    _GrpcClientPool,
    _pack_inference_tensor,
    _parse_raw_output,
    close_triton_grpc_client,
    get_triton_grpc_client,
)
from src.infrastructure.storage.torchscript_sanity import assert_triton_probability


@pytest.fixture(autouse=True)
def _reset_triton_pool():
    _GrpcClientPool.client = None
    yield
    _GrpcClientPool.client = None


@pytest.fixture(autouse=True)
def _restore_triton_asyncio_wait_for():
    real_wait_for = asyncio_tasks.wait_for
    asyncio.wait_for = real_wait_for
    triton_grpc_module.asyncio.wait_for = real_wait_for
    yield


class _FakeResult:
    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def as_numpy(self, _name: str):
        return self._arr


def test_parse_raw_output_clamps_probability():
    assert _parse_raw_output(_FakeResult(np.array([1.5], dtype=np.float32))) == 1.0
    assert _parse_raw_output(_FakeResult(np.array([0.42], dtype=np.float32))) == pytest.approx(0.42)
    assert _parse_raw_output(_FakeResult(None)) == 0.5


def test_assert_triton_probability_bounds():
    assert assert_triton_probability(0.72, model_name="RDBEAR") == 0.72
    with pytest.raises(RuntimeError, match="NaN"):
        assert_triton_probability(float("nan"), model_name="RDBEAR")
    with pytest.raises(RuntimeError, match="fora"):
        assert_triton_probability(1.05, model_name="RDBEAR")


def test_pack_inference_tensor_makes_contiguous_batch():
    tensor = np.arange(48 * 34, dtype=np.float32).reshape(48, 34)
    tensor.flags.writeable = False
    packed = _pack_inference_tensor(tensor)
    assert packed.flags["C_CONTIGUOUS"]
    assert packed.shape == (1, 48, 34)


def test_infer_timeout_default_is_850ms():
    assert pytest.approx(0.85) == _INFER_TIMEOUT_SEC


@pytest.mark.asyncio
async def test_triton_grpc_client_channel_property():
    client = TritonGrpcClient()
    channel = MagicMock()
    client._channel = channel
    assert client.channel is channel


@pytest.mark.asyncio
async def test_triton_grpc_client_infer_2d_tensor():
    client = TritonGrpcClient()
    mock_infer = AsyncMock(return_value=_FakeResult(np.array([0.33], dtype=np.float32)))
    client._infer = MagicMock()
    client._infer.infer = mock_infer
    with (
        patch("src.infrastructure.inference.triton_grpc_client.grpc_aio.InferInput") as infer_in,
        patch("src.infrastructure.inference.triton_grpc_client.grpc_aio.InferRequestedOutput"),
    ):
        infer_in.return_value.set_data_from_numpy = MagicMock()
        prob = await client.infer_symbol("RDBEAR", np.zeros((4, 34), dtype=np.float32))
    assert prob == pytest.approx(0.33)


@pytest.mark.asyncio
async def test_triton_grpc_client_concurrent_infer():
    client = TritonGrpcClient()
    mock_infer = AsyncMock(
        side_effect=[
            _FakeResult(np.array([0.61], dtype=np.float32)),
            _FakeResult(np.array([0.39], dtype=np.float32)),
        ]
    )
    client._infer = MagicMock()
    client._infer.infer = mock_infer
    client._channel = MagicMock()
    tensors = {
        "RDBEAR": np.zeros((1, 48, 34), dtype=np.float32),
        "RDBULL": np.zeros((1, 48, 34), dtype=np.float32),
    }
    probs = await client.infer_symbols_concurrent(tensors)
    assert probs["RDBEAR"] == pytest.approx(0.61)
    assert probs["RDBULL"] == pytest.approx(0.39)
    assert mock_infer.await_count == 2


@pytest.mark.asyncio
async def test_get_triton_grpc_client_singleton():
    with patch.object(TritonGrpcClient, "connect", AsyncMock()):
        first = await get_triton_grpc_client("localhost:8001")
        second = await get_triton_grpc_client("localhost:8001")
        assert first is second
    await close_triton_grpc_client()
    assert _GrpcClientPool.client is None


@pytest.mark.asyncio
async def test_triton_grpc_client_close_clears_channel():
    client = TritonGrpcClient()
    client._channel = MagicMock()
    client._channel.close = AsyncMock()
    client._infer = MagicMock()
    client._infer.close = AsyncMock()
    await client.close()
    assert client._channel is None
    assert client._infer is None


@pytest.mark.asyncio
async def test_triton_grpc_client_close_channel_only():
    client = TritonGrpcClient()
    channel = MagicMock()
    channel.close = AsyncMock()
    client._channel = channel
    client._infer = None
    await client._close_unlocked()
    channel.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_triton_grpc_client_connect_skips_when_already_connected():
    client = TritonGrpcClient()
    client._channel = MagicMock()
    client._url = "localhost:8001"
    with patch("src.infrastructure.inference.triton_grpc_client.grpc.aio.insecure_channel") as ctor:
        await client.connect("localhost:8001")
    ctor.assert_not_called()


@pytest.mark.asyncio
async def test_triton_grpc_client_infer_not_connected():
    client = TritonGrpcClient()
    with pytest.raises(RuntimeError, match="nao conectado"):
        await client.infer_symbol("RDBEAR", np.zeros((1, 4, 34), dtype=np.float32))


async def _timeout_wait_for(_coro, timeout=None):
    raise TimeoutError()


@pytest.mark.asyncio
async def test_triton_grpc_client_infer_timeout():
    client = TritonGrpcClient()
    client._infer = MagicMock()
    client._infer.infer = AsyncMock(return_value=_FakeResult(np.array([0.5], dtype=np.float32)))
    with (
        patch("src.infrastructure.inference.triton_grpc_client.grpc_aio.InferInput"),
        patch("src.infrastructure.inference.triton_grpc_client.grpc_aio.InferRequestedOutput"),
        patch.object(triton_grpc_module.asyncio, "wait_for", new=_timeout_wait_for),
        pytest.raises(triton_grpc_module.TritonInferenceTimeout, match=r"0\.850s"),
    ):
        await client.infer_symbol("RDBEAR", np.zeros((1, 4, 34), dtype=np.float32))


@pytest.mark.asyncio
async def test_triton_grpc_client_batch_infer_timeout():
    client = TritonGrpcClient()
    client._infer = MagicMock()
    client._infer.infer = AsyncMock(return_value=_FakeResult(np.array([0.5], dtype=np.float32)))
    tensors = {f"R_{i}": np.zeros((1, 48, 34), dtype=np.float32) for i in (10, 25, 50, 75, 100)}
    with (
        patch("src.infrastructure.inference.triton_grpc_client.grpc_aio.InferInput"),
        patch("src.infrastructure.inference.triton_grpc_client.grpc_aio.InferRequestedOutput"),
        patch.object(triton_grpc_module.asyncio, "wait_for", new=_timeout_wait_for),
        pytest.raises(triton_grpc_module.TritonInferenceTimeout, match=r"batch infer timeout"),
    ):
        await client.infer_symbols_concurrent(tensors)


@pytest.mark.asyncio
async def test_triton_grpc_client_infer_raises_server_exception():
    client = TritonGrpcClient()
    client._infer = MagicMock()
    client._infer.infer = AsyncMock(side_effect=InferenceServerException("fail"))
    with (
        patch("src.infrastructure.inference.triton_grpc_client.grpc_aio.InferInput"),
        patch("src.infrastructure.inference.triton_grpc_client.grpc_aio.InferRequestedOutput"),
        pytest.raises(InferenceServerException),
    ):
        await client.infer_symbol("RDBEAR", np.zeros((1, 4, 34), dtype=np.float32))


@pytest.mark.asyncio
async def test_triton_grpc_client_infer_empty_batch_returns():
    client = TritonGrpcClient()
    assert await client.infer_symbols_concurrent({}) == {}


def test_channel_options_with_keepalive():
    fake_keepalive = MagicMock()
    fake_keepalive.keepalive_time_ms = 120000
    fake_keepalive.keepalive_timeout_ms = 20000
    fake_keepalive.keepalive_permit_without_calls = True
    fake_keepalive.http2_max_pings_without_data = 0
    with patch("src.infrastructure.inference.triton_grpc_client.KeepAliveOptions", return_value=fake_keepalive):
        opts = _channel_options()
    assert ("grpc.keepalive_time_ms", 120000) in opts
    assert ("grpc.max_send_message_length", _MAX_MSG) in opts


@pytest.mark.asyncio
async def test_triton_grpc_client_connect_opens_channel():
    client = TritonGrpcClient()
    channel = MagicMock()
    infer_client = MagicMock()
    with (
        patch("src.infrastructure.inference.triton_grpc_client.grpc.aio.insecure_channel", return_value=channel),
        patch(
            "src.infrastructure.inference.triton_grpc_client.grpc_aio.InferenceServerClient.__new__",
            return_value=infer_client,
        ),
        patch("src.infrastructure.inference.triton_grpc_client._attach_channel", return_value=infer_client) as attach,
    ):
        await client.connect("localhost:8001")
    attach.assert_called_once()
    assert client._url == "localhost:8001"


def test_channel_options_without_keepalive():
    with patch("src.infrastructure.inference.triton_grpc_client.KeepAliveOptions", None):
        opts = _channel_options()
    assert ("grpc.max_send_message_length", _MAX_MSG) in opts


def test_parse_raw_output_nan_and_empty():
    nan = MagicMock()
    nan.as_numpy.return_value = np.array([float("nan")], dtype=np.float32)
    assert _parse_raw_output(nan) == 0.5
    empty = MagicMock()
    empty.as_numpy.return_value = np.array([], dtype=np.float32)
    assert _parse_raw_output(empty) == 0.5


def test_attach_channel_real():
    client = MagicMock()
    channel = MagicMock()
    with patch("src.infrastructure.inference.triton_grpc_client.InferenceServerClientBase.__init__") as mock_init:
        res = _attach_channel(client, channel)
    mock_init.assert_called_once_with(client)
    assert res._channel is channel
    assert res._verbose is False
