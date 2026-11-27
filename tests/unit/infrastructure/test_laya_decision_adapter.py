"""Testes unitarios para LayaDecisionAdapter e sanitizacao de settings."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from src.domain.models.market_data import Candle
from src.domain.models.trade import TradeDirection
from src.infrastructure.inference.laya_decision_adapter import (
    LayaDecisionAdapter,
    _compact_trading_settings,
    _extract_choice_probs,
    sanitize_settings_context,
)
from src.infrastructure.inference.laya_narrative import build_technical_narrative


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

    compact = _compact_trading_settings(
        {
            "anchor": "R_10",
            "infra": {"redis": {"url": "redis://localhost"}},
            "api_config": {"rest_base_url": "https://api.derivws.com"},
            "logging": {"level": "INFO"},
            "telemetry": {"port": 9100},
            "orchestrator": {"cycle_interval_seconds": 300},
            "data_handler": {"history_bars": 120},
            "risk_management": {"params": {"min_stake_pct": 0.01}},
        }
    )
    assert compact == {"mode": "demo", "duration_m": 5}


def test_build_technical_narrative_covers_all_regimes():
    candles = _sample_candles()
    bull = build_technical_narrative(
        "R_10",
        {
            "close": 105.0,
            "ema_9": 104.0,
            "ema_20": 102.0,
            "ema_50": 100.0,
            "ema_spread_atr": 1.0,
            "ema20_slope": 0.2,
            "adx_14": 26.0,
            "di_diff_14": 5.0,
            "hurst_14": 0.65,
            "macd_hist": 0.5,
            "macd_hist_delta": 0.1,
            "rsi_7": 72.0,
            "rsi_14": 70.0,
            "stoch_k": 85.0,
            "cci_14": 120.0,
            "bollinger_pct_b": 0.9,
            "atr_14": 2.0,
            "streak": 2.0,
            "body_atr": 0.5,
        },
        candles,
    )
    assert (
        "bullish (rising, price above EMA9/EMA20)" in bull
        and "strong trend" in bull
        and "overbought" in bull
        and "near upper band" in bull
    )
    bear = build_technical_narrative(
        "R_10",
        {
            "close": 95.0,
            "ema_9": 97.0,
            "ema_20": 98.0,
            "ema_50": 100.0,
            "ema20_slope": -0.2,
            "adx_14": 21.0,
            "di_diff_14": -4.0,
            "hurst_14": 0.40,
            "macd_hist": -0.5,
            "macd_hist_delta": -0.1,
            "rsi_14": 30.0,
            "bollinger_pct_b": 0.1,
            "atr_ratio": 1.20,
        },
        candles,
    )
    assert (
        "bearish (falling, price below EMA9/EMA20)" in bear
        and "moderate trend" in bear
        and "oversold" in bear
        and "near lower band" in bear
        and "expanding" in bear
    )
    neutral = build_technical_narrative(
        "R_10",
        {
            "close": 99.0,
            "ema_9": 100.0,
            "ema_20": 98.0,
            "ema_50": 97.0,
            "ema20_slope": -0.1,
            "macd_hist": 0.2,
            "macd_hist_delta": -0.1,
            "atr_ratio": 0.80,
        },
        candles,
    )
    assert (
        "mixed/transition (falling, price between EMA9/EMA20)" in neutral
        and "weak ranging" in neutral
        and "bullish fading" in neutral
        and "neutral" in neutral
        and "mid-band" in neutral
        and "compressed" in neutral
    )


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

        def system_one(self, state, questions, **kwargs):
            self.last_state = (json.loads(state), questions)
            return {
                "usage": {"truncated": False, "state_tokens": 100, "state_tokens_dropped": 0},
                "answers": {
                    "direction": {
                        "choice": "PUT",
                        "probabilities": {"CALL": 0.30, "PUT": 0.70},
                        "answer_confidence": 0.70,
                    }
                },
            }

    fake = _FakeAgent()
    monkeypatch.setattr(
        "src.infrastructure.inference.laya_decision_adapter.encode_laya_context",
        lambda agent, state, questions: (json.dumps(state), 512, "test"),
    )
    adapter = LayaDecisionAdapter(agent=fake)
    dec = await adapter.decide(
        "R_10", _sample_candles(), {"ema_20": 100.0, "ema_50": 101.0}, {"anchor": "R_10", "api_key": "x"}
    )
    assert dec.direction is TradeDirection.PUT
    assert dec.probability == pytest.approx(0.70)
    assert "mode" in fake.last_state[0]["settings"]
    assert len(fake.last_state[0]["recent_candles"]) == 5

    # Inferencia simetrica neutraliza vies posicional e desempata por EMA20 >= EMA50
    class _BiasedAgent:
        def system_one(self, _state, questions, **kwargs):
            first = next(iter(questions["direction"]["criteria"]))
            probs = {"CALL": 0.60, "PUT": 0.40} if first == "CALL" else {"CALL": 0.40, "PUT": 0.60}
            return {
                "usage": {"truncated": False, "state_tokens": 100, "state_tokens_dropped": 0},
                "answers": {"direction": {"probabilities": probs, "answer_confidence": 0.60}},
            }

    monkeypatch.setattr(
        "src.infrastructure.inference.laya_decision_adapter.encode_laya_context",
        lambda _a, state, _q: (json.dumps(state), 512, "sym"),
    )
    sym_adapter = LayaDecisionAdapter(agent=_BiasedAgent())
    sym_dec = await sym_adapter.decide("R_10", _sample_candles(), {"ema_20": 99.0, "ema_50": 101.0}, {})
    assert sym_dec.direction is TradeDirection.PUT
    assert sym_dec.prob_call == pytest.approx(0.5) and sym_dec.prob_put == pytest.approx(0.5)

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
