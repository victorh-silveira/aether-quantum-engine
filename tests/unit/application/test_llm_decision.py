from unittest.mock import patch

import pytest

from src.application.services.llm import llm_decision as llm


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


@pytest.mark.asyncio
async def test_get_decision_ok(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    with patch.object(llm, "_sync_generate", return_value="PUT"):
        out = await llm.get_decision("", "gemini-3-flash", "p", 10.0)
    assert out == ("PUT", True, "PUT")


@pytest.mark.asyncio
async def test_get_decision_retries_on_invalid_token(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    seq = iter(["invalid", "CALL"])

    def side(*_a, **_k):
        return next(seq)

    with patch.object(llm, "_sync_generate", side_effect=side):
        out = await llm.get_decision("", "gemini-3-flash", "p", 10.0, safety_retry_attempts=1)
    assert out == ("CALL", True, "CALL")


@pytest.mark.asyncio
async def test_get_decision_wait_is_invalid_and_continues_retries(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    seq = iter(["WAIT", "CALL"])
    with patch.object(llm, "_sync_generate", side_effect=lambda *a, **k: next(seq)):
        out = await llm.get_decision("", "gem", "p", 10.0, safety_retry_attempts=1)
    assert out == ("CALL", True, "CALL")


@pytest.mark.asyncio
async def test_get_decision_unexpected_response_logs_and_retries(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    seq = iter(["something else", "PUT"])
    with patch.object(llm, "_sync_generate", side_effect=lambda *a, **k: next(seq)):
        out = await llm.get_decision("", "gem", "p", 10.0, safety_retry_attempts=1)
    assert out == ("PUT", True, "PUT")


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


@pytest.mark.asyncio
async def test_get_decision_switches_to_fallback_on_transient_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    calls: list[str] = []

    async def fake_attempt(**kwargs):
        calls.append(kwargs["model_name"])
        if kwargs["model_name"] == "gemini-pro":
            return (None, False, "", Exception("503 UNAVAILABLE"))
        return ("PUT", True, "PUT", None)

    with patch.object(llm, "_attempt_generate", side_effect=fake_attempt):
        out = await llm.get_decision(
            "",
            "gemini-pro",
            "p",
            20.0,
            safety_retry_attempts=1,
            fallback_model="gemini-flash",
        )
    assert out == ("PUT", True, "PUT")
    assert calls[0] == "gemini-pro"
    assert "gemini-flash" in calls
