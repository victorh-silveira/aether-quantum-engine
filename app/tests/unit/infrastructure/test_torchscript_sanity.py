from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
from torch import nn

from src.application.services.deep_learning.dl_features import FEATURE_DIM
from src.infrastructure.storage.torchscript_sanity import (
    assert_triton_probability,
    validate_manifest_schema,
    verify_torchscript_artifact,
    verify_torchscript_artifact_async,
    verify_triton_stressed_inference_async,
)
from src.infrastructure.storage.torchscript_sanity_probes import build_sanity_probe_tensors


class _TinyNet(nn.Module):
    def __init__(self, feature_dim: int = FEATURE_DIM):
        super().__init__()
        self.fc = nn.Linear(feature_dim, 1)

    def forward(self, x):
        return self.fc(x[:, -1, :])


def _trace_model(path, *, lookback=48, feature_dim=FEATURE_DIM):
    model = _TinyNet(feature_dim)
    model.eval()
    example = torch.zeros(1, lookback, feature_dim)
    traced = torch.jit.trace(model, example, strict=False)
    traced.save(str(path))


def test_build_sanity_probe_tensors_count():
    probes = build_sanity_probe_tensors(48, FEATURE_DIM)
    assert len(probes) == 6
    assert probes[2][0] == "pos_extreme"
    assert probes[2][1][0, 0, 0].item() == 4.0
    stressed = probes[5]
    assert stressed[0] == "stressed_regime"
    assert stressed[1][0, 0, 5].item() == pytest.approx(0.99)


def test_validate_manifest_schema_lookback_mismatch():
    with pytest.raises(RuntimeError, match="lookback"):
        validate_manifest_schema(
            {"feature_dim": FEATURE_DIM, "lookback": 32},
            lookback=48,
            feature_dim=FEATURE_DIM,
        )


def test_validate_manifest_schema_norm_std_length():
    with pytest.raises(RuntimeError, match="norm_std"):
        validate_manifest_schema(
            {"norm_std": [1.0] * (FEATURE_DIM - 1)},
            lookback=48,
            feature_dim=FEATURE_DIM,
        )


def test_assert_triton_probability_rejects_inf():
    with pytest.raises(RuntimeError, match="NaN"):
        assert_triton_probability(float("inf"), model_name="R_50")


@pytest.mark.asyncio
async def test_verify_triton_stressed_inference_empty_symbols():
    await verify_triton_stressed_inference_async({}, [], lookback=48, feature_dim=FEATURE_DIM)


@pytest.mark.asyncio
async def test_verify_triton_stressed_inference_missing_response():
    cfg = {"infra": {"triton": {"enabled": True, "grpc_url": "localhost:8001"}}}
    mock_client = MagicMock()
    mock_client.infer_symbols_concurrent = AsyncMock(return_value={})
    with (
        patch(
            "src.infrastructure.storage.torchscript_sanity.get_triton_grpc_client",
            AsyncMock(return_value=mock_client),
        ),
        pytest.raises(RuntimeError, match="sem resposta"),
    ):
        await verify_triton_stressed_inference_async(cfg, ["R_10"], lookback=48, feature_dim=FEATURE_DIM)


@pytest.mark.asyncio
async def test_verify_triton_stressed_inference_async():
    cfg = {"infra": {"triton": {"enabled": True, "grpc_url": "localhost:8001"}}}
    mock_client = MagicMock()
    mock_client.infer_symbols_concurrent = AsyncMock(return_value={"R_10": 0.55})
    with patch(
        "src.infrastructure.storage.torchscript_sanity.get_triton_grpc_client",
        AsyncMock(return_value=mock_client),
    ):
        await verify_triton_stressed_inference_async(cfg, ["R_10"], lookback=48, feature_dim=FEATURE_DIM)
    mock_client.infer_symbols_concurrent.assert_awaited_once()


@pytest.mark.asyncio
async def test_verify_triton_stressed_inference_fail_fast_on_oob():
    cfg = {"infra": {"triton": {"enabled": True, "grpc_url": "localhost:8001"}}}
    mock_client = MagicMock()
    mock_client.infer_symbols_concurrent = AsyncMock(return_value={"R_10": 1.2})
    with (
        patch(
            "src.infrastructure.storage.torchscript_sanity.get_triton_grpc_client",
            AsyncMock(return_value=mock_client),
        ),
        pytest.raises(RuntimeError, match="fora"),
    ):
        await verify_triton_stressed_inference_async(cfg, ["R_10"], lookback=48, feature_dim=FEATURE_DIM)


@pytest.mark.asyncio
async def test_verify_torchscript_artifact_async(tmp_path):
    path = tmp_path / "async_ts.pt"
    _trace_model(path)
    await verify_torchscript_artifact_async(path, lookback=48, feature_dim=FEATURE_DIM)


def test_validate_manifest_schema_mismatch_raises():
    with pytest.raises(RuntimeError, match="feature_dim"):
        validate_manifest_schema(
            {"feature_dim": FEATURE_DIM - 1},
            lookback=48,
            feature_dim=FEATURE_DIM,
        )


def test_validate_manifest_schema_norm_length():
    with pytest.raises(RuntimeError, match="norm_mean"):
        validate_manifest_schema(
            {"norm_mean": [0.0] * (FEATURE_DIM - 1)},
            lookback=48,
            feature_dim=FEATURE_DIM,
        )


def test_verify_torchscript_artifact_success(tmp_path):
    path = tmp_path / "m_ts.pt"
    _trace_model(path)
    verify_torchscript_artifact(path, lookback=48, feature_dim=FEATURE_DIM)


def test_verify_torchscript_nan_raises(tmp_path):
    path = tmp_path / "bad_ts.pt"

    class _NanNet(nn.Module):
        def forward(self, _x):
            return torch.tensor([[float("nan")]])

    traced = torch.jit.trace(_NanNet(), torch.zeros(1, 48, FEATURE_DIM), strict=False)
    traced.save(str(path))
    with pytest.raises(RuntimeError, match="NaN"):
        verify_torchscript_artifact(path, lookback=48, feature_dim=FEATURE_DIM)


def test_verify_torchscript_missing_file(tmp_path):
    with pytest.raises(RuntimeError, match="ausente"):
        verify_torchscript_artifact(tmp_path / "missing.pt", lookback=48, feature_dim=FEATURE_DIM)


def test_verify_torchscript_tuple_output(tmp_path):
    path = tmp_path / "tuple_ts.pt"

    class _TupleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(FEATURE_DIM, 1)

        def forward(self, x):
            return (self.fc(x[:, -1, :]),)

    traced = torch.jit.trace(_TupleNet(), torch.zeros(1, 48, FEATURE_DIM), strict=False)
    traced.save(str(path))
    verify_torchscript_artifact(path, lookback=48, feature_dim=FEATURE_DIM)


def test_verify_torchscript_invalid_type_raises(tmp_path):
    path = tmp_path / "bad_type.pt"
    path.write_bytes(b"x")
    model = MagicMock()
    model.eval.return_value = None
    model.return_value = 1
    with patch("torch.jit.load", return_value=model), pytest.raises(RuntimeError, match="tipo invalido"):
        verify_torchscript_artifact(path, lookback=48, feature_dim=FEATURE_DIM)


def test_verify_torchscript_invalid_shape_raises(tmp_path):
    path = tmp_path / "bad_shape.pt"

    class _EmptyNet(nn.Module):
        def forward(self, _x):
            return torch.zeros(0)

    traced = torch.jit.trace(_EmptyNet(), torch.zeros(1, 48, FEATURE_DIM), strict=False)
    traced.save(str(path))
    with pytest.raises(RuntimeError, match="shape invalida"):
        verify_torchscript_artifact(path, lookback=48, feature_dim=FEATURE_DIM)


def test_verify_torchscript_with_valid_manifest(tmp_path):
    path = tmp_path / "ok_ts.pt"
    _trace_model(path)
    manifest = {
        "feature_dim": FEATURE_DIM,
        "lookback": 48,
        "norm_mean": [0.0] * FEATURE_DIM,
        "norm_std": [1.0] * FEATURE_DIM,
    }
    verify_torchscript_artifact(
        path,
        lookback=48,
        feature_dim=FEATURE_DIM,
        manifest=manifest,
    )
