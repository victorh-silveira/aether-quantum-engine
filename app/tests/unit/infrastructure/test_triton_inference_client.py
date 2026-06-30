from unittest.mock import patch

import numpy as np
import pytest

from src.infrastructure.inference.triton_inference_client import (
    _parse_raw_output,
    reload_triton_repository,
    triton_http_url,
)


class _FakeResult:
    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def as_numpy(self, _name: str):
        return self._arr


def test_parse_raw_output_clamps_probability():
    assert _parse_raw_output(_FakeResult(np.array([1.5], dtype=np.float32))) == 1.0
    assert _parse_raw_output(_FakeResult(np.array([0.42], dtype=np.float32))) == pytest.approx(0.42)
    assert _parse_raw_output(_FakeResult(None)) == 0.5


def test_triton_http_url_from_config():
    cfg = {"infra": {"triton": {"http_url": "http://triton:8000"}}}
    assert triton_http_url(cfg) == "http://triton:8000"


@pytest.mark.asyncio
async def test_reload_triton_repository_success():
    cfg = {"infra": {"triton": {"enabled": True, "http_url": "http://localhost:8000"}}}
    with patch(
        "src.infrastructure.inference.triton_inference_client.post_triton_repository_reload",
        return_value=[{"name": "R_10"}],
    ):
        ok = await reload_triton_repository(cfg)
    assert ok is True


@pytest.mark.asyncio
async def test_reload_triton_repository_disabled():
    ok = await reload_triton_repository({"infra": {"triton": {"enabled": False}}})
    assert ok is False


@pytest.mark.asyncio
async def test_reload_triton_repository_http_error():
    cfg = {"infra": {"triton": {"enabled": True, "http_url": "http://localhost:8000"}}}
    with patch(
        "src.infrastructure.inference.triton_inference_client.post_triton_repository_reload",
        side_effect=OSError("connection refused"),
    ):
        ok = await reload_triton_repository(cfg)
    assert ok is False
