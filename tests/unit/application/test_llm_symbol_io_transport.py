from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.llm.llm_symbol_io import request_llm_payload


@pytest.mark.asyncio
async def test_request_llm_payload_async_timeout_fallback_put():
    orch = MagicMock()
    orch.logger = MagicMock()
    runtime = {
        "base_url": "",
        "model": "m",
        "timeout": 5.0,
        "num_predict": 150,
        "keep_alive": "60m",
        "parse_retry_attempts": 1,
    }

    async def wf_timeout(coro, timeout=None):
        coro.close()
        raise TimeoutError()

    with patch("src.application.services.llm.llm_symbol_io.asyncio.wait_for", wf_timeout):
        payload = await request_llm_payload(
            orch,
            "1HZ75V",
            runtime,
            "MTF=B/B/B SYM=1HZ75V",
            system="sys",
        )
    assert payload.get("_llm_call_failed") is True
    assert payload.get("_direction_normalized") is None


@pytest.mark.asyncio
async def test_request_llm_payload_async_timeout_prompt_vazio_fallback_put():
    orch = MagicMock()
    orch.logger = MagicMock()
    runtime = {
        "base_url": "",
        "model": "m",
        "timeout": 5.0,
        "num_predict": 150,
        "keep_alive": "60m",
        "parse_retry_attempts": 1,
    }

    async def wf_timeout(coro, timeout=None):
        coro.close()
        raise TimeoutError()

    with patch("src.application.services.llm.llm_symbol_io.asyncio.wait_for", wf_timeout):
        payload = await request_llm_payload(orch, "1HZ75V", runtime, "", system="sys")
    assert payload.get("_llm_call_failed") is True
    assert payload.get("note") == "llm_timeout"
    assert payload.get("_direction_normalized") is None


@pytest.mark.asyncio
async def test_request_llm_payload_wait_api_fallback_put_majoria_b():
    orch = MagicMock()
    orch.logger = MagicMock()
    runtime = {
        "base_url": "",
        "model": "m",
        "timeout": 5.0,
        "num_predict": 150,
        "keep_alive": "60m",
        "parse_retry_attempts": 2,
        "gemini_api_key": "x",
        "generation_config": {},
    }
    with patch(
        "src.application.services.llm.llm_symbol_io.get_decision",
        new_callable=AsyncMock,
        return_value=("WAIT", False, "WAIT"),
    ):
        payload = await request_llm_payload(
            orch,
            "1HZ75V",
            runtime,
            "mtf=na, RSI=50, MTF=B/B/B, SYM=1HZ75V",
            system="sys",
        )
    assert not payload.get("_llm_call_failed")
    assert payload.get("_direction_normalized") is None
