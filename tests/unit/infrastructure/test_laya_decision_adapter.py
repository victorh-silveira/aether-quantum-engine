"""Testes unitarios para LayaDecisionAdapter e sanitizacao de settings."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from src.domain.models.market_data import Candle
from src.domain.models.trade import TradeDirection
from src.infrastructure.inference.laya_decision_adapter import (
    LayaDecisionAdapter,
    _extract_choice_probs,
    sanitize_settings_context,
)


def _sample_candles() -> list[Candle]:
    return [
        Candle(
            "R_10",
            100.0 + i,
            101.0 + i,
            99.0 + i,
            100.5 + i,
            datetime.fromtimestamp(1_764_000_000 + i * 300, tz=UTC),
            1_764_000_000 + i * 300,
        )
        for i in range(5)
    ]


def test_sanitize_settings_removes_sensitive_keys():
    cleaned = sanitize_settings_context(
        {
            "anchor": "R_10",
            "deriv_app_id": "secret123",
            "nested": {"pat_token": "abc", "symbols": ["R_10"], "dsn": "pg://secret"},
        }
    )
    assert cleaned["anchor"] == "R_10"
    assert "deriv_app_id" not in cleaned
    assert cleaned["nested"] == {"symbols": ["R_10"]}


def test_extract_choice_probs_handles_real_schema_and_ties():
    real_payload = {
        "answers": {
            "direction": {
                "choice": "CALL",
                "probabilities": {"CALL": 0.76, "PUT": 0.24},
                "answer_confidence": 0.76,
            }
        }
    }
    direction, p_c, p_p, conf = _extract_choice_probs(real_payload, {"ema_20": 10.0, "ema_50": 9.0})
    assert direction is TradeDirection.CALL
    assert p_c == pytest.approx(0.76)
    assert p_p == pytest.approx(0.24)
    assert conf == pytest.approx(0.76)

    # Fallback top-level e empate deterministico
    tie_payload = {
        "probabilities": {"direction": {"CALL": math.nan, "PUT": -1.0}},
        "confidence": {"direction": math.nan},
    }
    dir_put, p_c2, p_p2, conf2 = _extract_choice_probs(tie_payload, {"ema_20": 8.0, "ema_50": 9.0})
    assert dir_put is TradeDirection.PUT
    assert p_c2 == 0.5 and p_p2 == 0.5 and conf2 == 0.5

    zero_payload = {"answers": {"direction": {"probabilities": {"CALL": 1e-15, "PUT": 1e-15}}}}
    dir_call, p_c3, p_p3, _ = _extract_choice_probs(zero_payload, {"ema_20": 10.0, "ema_50": 10.0})
    assert dir_call is TradeDirection.CALL
    assert p_c3 == 0.5 and p_p3 == 0.5

    bad_probs = {"probabilities": {"direction": "invalid"}}
    dir_bad, p_c4, p_p4, _ = _extract_choice_probs(bad_probs, {"ema_20": 10.0, "ema_50": 9.0})
    assert dir_bad is TradeDirection.CALL
    assert p_c4 == 0.5 and p_p4 == 0.5


@pytest.mark.asyncio
async def test_laya_decision_adapter_decide_async(monkeypatch):
    class _FakeAgent:
        def __init__(self):
            self.last_state = None

        def system_one(self, state, questions):
            self.last_state = (state, questions)
            return {
                "answers": {
                    "direction": {
                        "choice": "PUT",
                        "probabilities": {"CALL": 0.30, "PUT": 0.70},
                        "answer_confidence": 0.70,
                    }
                }
            }

    fake = _FakeAgent()
    adapter = LayaDecisionAdapter(agent=fake)
    dec = await adapter.decide(
        "R_10", _sample_candles(), {"ema_20": 100.0, "ema_50": 101.0}, {"anchor": "R_10", "api_key": "x"}
    )
    assert dec.direction is TradeDirection.PUT
    assert dec.probability == pytest.approx(0.70)
    assert "api_key" not in fake.last_state[0]["settings"]
    assert len(fake.last_state[0]["recent_candles"]) == 5

    # Testa _ensure_agent lazy load via monkeypatch
    called = {}
    monkeypatch.setitem(
        __import__("sys").modules,
        "laya",
        SimpleNamespace(
            load=lambda mid, device=None, vegaml=None: (called.setdefault("args", (mid, device, vegaml)), fake)[1]
        ),
    )
    monkeypatch.setitem(
        __import__("sys").modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    )
    lazy_adapter = LayaDecisionAdapter(model_id="convaiinnovations/laya", device="auto")
    assert lazy_adapter._ensure_agent() is fake
    assert called["args"] == ("convaiinnovations/laya", "cpu", False)
