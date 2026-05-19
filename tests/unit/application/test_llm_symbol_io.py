import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import src.application.services.llm.llm_symbol_io as llm_io
from src.application.services.llm.llm_symbol_io import request_llm_payload


@pytest.mark.asyncio
async def test_request_llm_payload_propaga_call_normalizado():
    orch = MagicMock()
    runtime = {
        "base_url": "http://x",
        "model": "m",
        "timeout": 1.0,
        "num_predict": 64,
        "llm_async_outer_seconds": 30.0,
    }
    with patch(
        "src.application.services.llm.llm_symbol_io.get_decision",
        new_callable=AsyncMock,
        return_value=("CALL", True, "CALL"),
    ) as gen:
        payload = await request_llm_payload(orch, "1HZ75V", runtime, "p", system="s")
    gen.assert_awaited_once()
    assert "allow_payload_fallback" not in gen.await_args.kwargs
    assert payload["_direction_normalized"] == "CALL"
    assert payload["_conviction_normalized"] == pytest.approx(0.99, abs=1e-9)
    assert payload["_llm_direction_from_api"] is True
    assert payload["_llm_raw_chars"] == 4


@pytest.mark.asyncio
async def test_request_llm_payload_sem_token_valido_fallback_put():
    orch = MagicMock()
    runtime = {
        "base_url": "http://x",
        "model": "m",
        "timeout": 1.0,
        "num_predict": 64,
        "llm_async_outer_seconds": 30.0,
    }
    with patch(
        "src.application.services.llm.llm_symbol_io.get_decision",
        new_callable=AsyncMock,
        return_value=("WAIT", True, "WAIT"),
    ):
        payload = await request_llm_payload(orch, "1HZ75V", runtime, "p", system="s")
    assert payload["_direction_normalized"] is None
    assert payload["_llm_direction_from_api"] is True


@pytest.mark.asyncio
async def test_request_llm_payload_wait_api_usa_cf30_fallback():
    orch = MagicMock()
    runtime = {
        "base_url": "http://x",
        "model": "m",
        "timeout": 1.0,
        "num_predict": 64,
        "llm_async_outer_seconds": 30.0,
        "llm_wait_fallback_mode": "macro_cf30",
        "llm_retry_attempts": 0,
    }
    pr = "MTF=A/A/B || cf30_5=fC|cf5_1=divM5M1"
    with patch(
        "src.application.services.llm.llm_symbol_io.get_decision",
        new_callable=AsyncMock,
        return_value=("WAIT", True, "WAIT"),
    ):
        payload = await request_llm_payload(orch, "1HZ75V", runtime, pr, system="s")
    assert payload["_direction_normalized"] is None
    assert payload["_llm_direction_from_api"] is True


@pytest.mark.asyncio
async def test_request_llm_payload_timeout_externo_fallback_call():
    orch = MagicMock()
    runtime = {
        "base_url": "http://x",
        "model": "m",
        "timeout": 10.0,
        "num_predict": 64,
        "llm_async_outer_seconds": 0.02,
    }

    async def slow(*_a, **_k):
        await asyncio.sleep(1.0)
        return ("CALL", True, "CALL")

    with patch(
        "src.application.services.llm.llm_symbol_io.get_decision",
        new_callable=AsyncMock,
        side_effect=slow,
    ):
        payload = await request_llm_payload(orch, "S", runtime, "MTF=A/A/A z", system="s")
    assert payload["_direction_normalized"] is None
    assert payload.get("_llm_call_failed") is True


@pytest.mark.asyncio
async def test_request_llm_payload_info_quando_http_ms_alto():
    orch = MagicMock()
    runtime = {
        "base_url": "http://x",
        "model": "m",
        "timeout": 10.0,
        "num_predict": 64,
        "llm_async_outer_seconds": 60.0,
    }
    seq = iter([0.0, 0.6])

    def fake_perf():
        return next(seq)

    with (
        patch.object(llm_io._logger, "debug") as debug_log,
        patch.object(llm_io.time, "perf_counter", side_effect=fake_perf),
        patch(
            "src.application.services.llm.llm_symbol_io.get_decision",
            new_callable=AsyncMock,
            return_value=("CALL", True, "CALL"),
        ),
    ):
        await request_llm_payload(orch, "S", runtime, "p", system="s", cycle_id=9)
    assert debug_log.called
    lat_call = [c for c in debug_log.call_args_list if "latencia elevada" in str(c.args[0])]
    assert lat_call
    args = lat_call[0].args
    rendered = args[0] % args[1:]
    assert "[C0009]" in rendered


@pytest.mark.asyncio
async def test_request_llm_payload_clusters_extraction():
    orch = MagicMock()
    runtime = {
        "base_url": "http://x",
        "model": "m",
        "timeout": 1.0,
        "num_predict": 64,
        "llm_async_outer_seconds": 30.0,
    }
    raw_response = "EURUSD: PUT | US_CLUSTER: PUT | EU_CLUSTER: CALL | Probabilidade: 85.5%"
    with patch(
        "src.application.services.llm.llm_symbol_io.get_decision",
        new_callable=AsyncMock,
        return_value=("PUT", True, raw_response),
    ):
        payload = await request_llm_payload(orch, "frxEURUSD", runtime, "p", system="s")

    assert payload["_direction_normalized"] == "PUT"
    assert payload["us_cluster"] == "PUT"
    assert payload["eu_cluster"] == "CALL"
    assert payload["_conviction_normalized"] == 0.855
