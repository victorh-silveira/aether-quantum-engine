from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.application.services.llm import llm_decision as llm


def _empty_resp():
    return SimpleNamespace(candidates=[SimpleNamespace(finish_reason="STOP")])


@pytest.mark.asyncio
async def test_get_decision_none_on_api_key_missing(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with patch.object(llm, "load_dotenv"):
        out = await llm.get_decision("", "gemini-3-flash", "p", 10.0, api_key="")
    assert out == (None, False, "")


@pytest.mark.asyncio
async def test_get_decision_none_on_timeout(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")

    async def _timeout(coro, **_k):
        if hasattr(coro, "close"):
            coro.close()
        raise TimeoutError()

    with patch("src.application.services.llm.llm_decision.asyncio.wait_for", side_effect=_timeout):
        out = await llm.get_decision("", "gemini-3-flash", "p", 0.1)
    assert out == (None, False, "")


_FULL = "EURUSD: PUT | US_CLUSTER: PUT | EU_CLUSTER: CALL | Probabilidade: 0.72"


@pytest.mark.asyncio
async def test_attempt_generate_logs_max_tokens(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    resp = SimpleNamespace(candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")])

    def fake_sync(*_a, **_k):
        return "", resp

    with patch.object(llm, "_sync_generate", side_effect=fake_sync):
        norm, ok, raw, err = await llm._attempt_generate(
            model_name="m",
            api_key="k",
            body="p",
            gen_cfg=None,
            deadline=10.0,
            log_cycle_id=1,
        )
    assert norm is None
    assert err is None
    assert raw == ""


@pytest.mark.asyncio
async def test_get_decision_ok(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    with patch.object(llm, "_sync_generate", return_value=(_FULL, _empty_resp())):
        out = await llm.get_decision("", "gemini-3-flash", "p", 10.0)
    assert out == ("PUT", True, _FULL)


@pytest.mark.asyncio
async def test_get_decision_retries_on_invalid_token(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    seq = iter(["invalid", _FULL])

    def side(*_a, **_k):
        return next(seq), _empty_resp()

    with patch.object(llm, "_sync_generate", side_effect=side):
        out = await llm.get_decision("", "gemini-3-flash", "p", 10.0, safety_retry_attempts=1)
    assert out[0] == "PUT"
    assert out[1] is True


@pytest.mark.asyncio
async def test_get_decision_wait_is_invalid_and_continues_retries(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    seq = iter(["WAIT", _FULL])
    with patch.object(llm, "_sync_generate", side_effect=lambda *a, **k: (next(seq), _empty_resp())):
        out = await llm.get_decision("", "gem", "p", 10.0, safety_retry_attempts=1)
    assert out[0] == "PUT"


@pytest.mark.asyncio
async def test_get_decision_unexpected_response_logs_and_retries(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    seq = iter(["MACRO_CONFLUENCIA indica misto", _FULL])
    with patch.object(llm, "_sync_generate", side_effect=lambda *a, **k: (next(seq), _empty_resp())):
        out = await llm.get_decision("", "gem", "p", 10.0, safety_retry_attempts=1)
    assert out[0] == "PUT"


def test_resolved_system_instruction_base_only():
    assert "Aether-Quantum-Engine" in llm._resolved_system_instruction(None)
    assert llm._resolved_system_instruction("") == llm.SOVEREIGN_SYSTEM


def test_resolved_system_instruction_appends_runtime():
    out = llm._resolved_system_instruction("extra ctx")
    assert "extra ctx" in out
    assert llm.SOVEREIGN_SYSTEM.split("\n")[0] in out


def test_is_retryable_gemini_error_detects_transient_codes():
    assert llm.is_retryable_gemini_error(Exception("504 DEADLINE_EXCEEDED"))
    assert llm.is_retryable_gemini_error(Exception("503 UNAVAILABLE"))
    assert not llm.is_retryable_gemini_error(Exception("400 INVALID_ARGUMENT"))
    assert not llm.is_retryable_gemini_error(Exception("404 NOT_FOUND"))


def test_is_invalid_model_gemini_error_detects_404():
    assert llm.is_invalid_model_gemini_error(Exception("404 NOT_FOUND"))
    assert not llm.is_invalid_model_gemini_error(Exception("503 UNAVAILABLE"))


@pytest.mark.asyncio
async def test_get_decision_switches_to_fallback_on_transient_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    calls: list[str] = []

    async def fake_attempt(**kwargs):
        calls.append(kwargs["model_name"])
        if kwargs["model_name"] == "gemini-pro":
            return (None, False, "", Exception("503 UNAVAILABLE"))
        return ("PUT", True, _FULL, None)

    with patch.object(llm, "_attempt_generate", side_effect=fake_attempt):
        out = await llm.get_decision(
            "",
            "gemini-pro",
            "p",
            20.0,
            safety_retry_attempts=1,
            fallback_model="gemini-flash",
        )
    assert out[0] == "PUT"
    assert calls[0] == "gemini-pro"
    assert "gemini-flash" in calls


_JSON_FULL = '{"EURUSD":"CALL","US_CLUSTER":"CALL","EU_CLUSTER":"PUT","Probabilidade":0.72}'


@pytest.mark.asyncio
async def test_get_decision_equal_primary_fallback_uses_default_lite(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    calls: list[str] = []

    async def fake_attempt(**kwargs):
        calls.append(kwargs["model_name"])
        if len(calls) == 1:
            return (None, False, "", Exception("503 UNAVAILABLE"))
        return ("CALL", True, _JSON_FULL, None)

    with patch.object(llm, "_attempt_generate", side_effect=fake_attempt):
        out = await llm.get_decision(
            "",
            llm.GEMINI_DEFAULT_MODEL,
            "p",
            20.0,
            safety_retry_attempts=1,
            fallback_model=llm.GEMINI_DEFAULT_MODEL,
        )
    assert out[0] == "CALL"
    assert calls[0] == llm.GEMINI_DEFAULT_MODEL
    assert calls[1] == llm.GEMINI_FALLBACK_MODEL


@pytest.mark.asyncio
async def test_get_decision_invalid_fallback_model_reverts_to_primary(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    calls: list[str] = []

    async def fake_attempt(**kwargs):
        calls.append(kwargs["model_name"])
        if kwargs["model_name"] == "fb-bad":
            return (None, False, "", Exception("404 NOT_FOUND"))
        if len([c for c in calls if c == "main"]) == 1:
            return (None, False, "", Exception("503 UNAVAILABLE"))
        return ("PUT", True, _JSON_FULL, None)

    with patch.object(llm, "_attempt_generate", side_effect=fake_attempt):
        out = await llm.get_decision(
            "",
            "main",
            "p",
            20.0,
            safety_retry_attempts=2,
            fallback_model="fb-bad",
        )
    assert out[0] == "PUT"
    assert "fb-bad" in calls
    assert calls[-1] == "main"
