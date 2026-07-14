from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from src.infrastructure.inference.triton_inference_client import (
    _parse_raw_output,
    _triton_wait_settings,
    reload_triton_repository,
    triton_http_url,
    triton_infer_timeout_seconds,
    wait_triton_models_stable,
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


def test_triton_wait_settings_defaults_when_chunk_invalid():
    assert _triton_wait_settings({"infra": {"triton": "bad"}}) == (25.0, 0.5)


def test_triton_infer_timeout_seconds_reads_bootstrap_and_cycle_defaults():
    cfg = {
        "infra": {
            "triton": {
                "infer_timeout_seconds": 0.9,
                "bootstrap_infer_timeout_seconds": 6.0,
            }
        }
    }
    assert triton_infer_timeout_seconds(cfg, bootstrap=False) == pytest.approx(0.9)
    assert triton_infer_timeout_seconds(cfg, bootstrap=True) == pytest.approx(6.0)
    assert triton_infer_timeout_seconds({}, bootstrap=True) == pytest.approx(5.0)
    assert triton_infer_timeout_seconds({}, bootstrap=False) == pytest.approx(0.85)
    assert triton_infer_timeout_seconds({"infra": {"triton": "bad"}}, bootstrap=False) == pytest.approx(0.85)


@pytest.mark.asyncio
async def test_reload_triton_repository_success():
    cfg = {"infra": {"triton": {"enabled": True, "http_url": "http://localhost:8000"}}}
    with patch(
        "src.infrastructure.inference.triton_inference_client.post_triton_repository_reload",
        return_value=[{"name": "RDBEAR"}],
    ):
        ok = await reload_triton_repository(cfg)
    assert ok is True


@pytest.mark.asyncio
async def test_reload_triton_repository_waits_for_symbols():
    cfg = {
        "infra": {
            "triton": {
                "enabled": True,
                "http_url": "http://localhost:8000",
                "wait_ready_seconds": 5.0,
                "poll_ready_seconds": 0.1,
                "repository_poll_seconds": 2.0,
            }
        }
    }
    with (
        patch(
            "src.infrastructure.inference.triton_inference_client.wait_triton_models_ready",
            return_value=True,
        ) as wait_mock,
        patch("src.infrastructure.inference.triton_inference_client.asyncio.sleep", new_callable=AsyncMock),
    ):
        ok = await reload_triton_repository(cfg, ["RDBEAR", "RDBULL"])
    assert ok is True
    assert wait_mock.call_count == 2


@pytest.mark.asyncio
async def test_wait_triton_models_stable_skips_settle_when_repo_unchanged():
    cfg = {"infra": {"triton": {"enabled": True, "http_url": "http://localhost:8000"}}}
    with (
        patch(
            "src.infrastructure.inference.triton_inference_client.wait_triton_models_ready",
            return_value=True,
        ) as wait_mock,
        patch(
            "src.infrastructure.inference.triton_inference_client.asyncio.sleep", new_callable=AsyncMock
        ) as sleep_mock,
    ):
        ok = await wait_triton_models_stable(cfg, ["RDBEAR"], repo_changed=False)
    assert ok is True
    wait_mock.assert_called_once()
    sleep_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_wait_triton_models_stable_disabled():
    ok = await wait_triton_models_stable({"infra": {"triton": {"enabled": False}}}, ["RDBEAR"])
    assert ok is False


@pytest.mark.asyncio
async def test_wait_triton_models_stable_empty_symbol_list():
    ok = await wait_triton_models_stable({"infra": {"triton": {"enabled": True}}}, [])
    assert ok is True


@pytest.mark.asyncio
async def test_wait_triton_models_stable_returns_false_when_not_ready():
    cfg = {"infra": {"triton": {"enabled": True, "http_url": "http://localhost:8000"}}}
    with patch(
        "src.infrastructure.inference.triton_inference_client.wait_triton_models_ready",
        return_value=False,
    ):
        ok = await wait_triton_models_stable(cfg, ["RDBEAR"], repo_changed=True)
    assert ok is False


@pytest.mark.asyncio
async def test_reload_triton_repository_wait_timeout():
    cfg = {"infra": {"triton": {"enabled": True, "http_url": "http://localhost:8000"}}}
    with patch(
        "src.infrastructure.inference.triton_inference_client.wait_triton_models_ready",
        return_value=False,
    ):
        ok = await reload_triton_repository(cfg, ["RDBEAR"])
    assert ok is False


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
