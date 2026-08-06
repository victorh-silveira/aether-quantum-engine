from unittest.mock import patch

import pytest

from src.application.services.deep_learning.dl_features import FEATURE_DIM
from src.infrastructure.inference.triton_inference_client import triton_http_url
from src.infrastructure.inference.triton_model_metadata import (
    fetch_triton_model_metadata_async,
    parse_triton_input_dims,
)
from src.infrastructure.storage.torchscript_sanity import (
    assert_triton_host_schema_aligned,
    verify_triton_schema_alignment_async,
)


def test_parse_triton_input_dims_from_http_payload():
    payload = {
        "name": "OTC_SPC",
        "inputs": [{"name": "INPUT__0", "datatype": "FP32", "shape": [-1, 48, 34]}],
    }
    feature_dim, lookback = parse_triton_input_dims(payload)
    assert feature_dim == 34
    assert lookback == 48


def test_parse_triton_input_dims_dynamic_lookback():
    feature_dim, lookback = parse_triton_input_dims({"inputs": [{"shape": [-1, -1, 34]}]})
    assert feature_dim == 34
    assert lookback is None


def test_parse_triton_input_dims_errors():
    with pytest.raises(RuntimeError, match="sem bloco inputs"):
        parse_triton_input_dims({})
    with pytest.raises(RuntimeError, match="inputs\\[0\\] invalido"):
        parse_triton_input_dims({"inputs": ["x"]})
    with pytest.raises(RuntimeError, match="shape invalido"):
        parse_triton_input_dims({"inputs": [{"shape": [48]}]})
    with pytest.raises(RuntimeError, match="feature_dim invalido"):
        parse_triton_input_dims({"inputs": [{"shape": [-1, 48, 0]}]})


def test_assert_triton_host_schema_aligned_ok():
    payload = {"inputs": [{"shape": [-1, 48, FEATURE_DIM]}]}
    assert_triton_host_schema_aligned(
        payload,
        host_feature_dim=FEATURE_DIM,
        host_lookback=48,
        model_name="OTC_SPC",
    )


def test_assert_triton_host_schema_aligned_feature_dim_mismatch():
    payload = {"inputs": [{"shape": [-1, 48, 19]}]}
    with pytest.raises(RuntimeError, match="feature_dim=19"):
        assert_triton_host_schema_aligned(
            payload,
            host_feature_dim=FEATURE_DIM,
            host_lookback=48,
            model_name="OTC_SPC",
        )


def test_assert_triton_host_schema_aligned_lookback_mismatch():
    payload = {"inputs": [{"shape": [-1, 32, FEATURE_DIM]}]}
    with pytest.raises(RuntimeError, match="lookback=32"):
        assert_triton_host_schema_aligned(
            payload,
            host_feature_dim=FEATURE_DIM,
            host_lookback=48,
            model_name="OTC_SPC",
        )


@pytest.mark.asyncio
async def test_verify_triton_schema_alignment_async():
    cfg = {"infra": {"triton": {"enabled": True, "http_url": "http://localhost:8000"}}}
    payload = {"inputs": [{"shape": [-1, 48, FEATURE_DIM]}]}
    with patch(
        "src.infrastructure.storage.torchscript_sanity.fetch_triton_model_metadata_async",
        return_value=payload,
    ) as fetch:
        await verify_triton_schema_alignment_async(
            cfg,
            "OTC_SPC",
            host_feature_dim=FEATURE_DIM,
            host_lookback=48,
        )
    fetch.assert_awaited_once_with(cfg, "OTC_SPC")


@pytest.mark.asyncio
async def test_fetch_triton_model_metadata_async_disabled():
    with pytest.raises(RuntimeError, match="desabilitado"):
        await fetch_triton_model_metadata_async({"infra": {"triton": {"enabled": False}}}, "OTC_SPC")


@pytest.mark.asyncio
async def test_fetch_triton_model_metadata_async_enabled():
    cfg = {"infra": {"triton": {"enabled": True, "http_url": "localhost:8000"}}}
    payload = {"inputs": [{"shape": [-1, 48, FEATURE_DIM]}]}
    with patch(
        "src.infrastructure.inference.triton_model_metadata.get_triton_model_metadata",
        return_value=payload,
    ) as fetch:
        out = await fetch_triton_model_metadata_async(cfg, "OTC_SPC")
    assert out == payload
    fetch.assert_called_once_with("http://localhost:8000", "OTC_SPC")


def test_triton_http_url_default(monkeypatch):
    monkeypatch.delenv("AETHER_TRITON_HTTP", raising=False)
    assert triton_http_url({}) == "http://localhost:8000"
