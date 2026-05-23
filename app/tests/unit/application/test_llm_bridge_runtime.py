from unittest.mock import MagicMock

import pytest

from src.application.services.llm import llm_bridge as bridge


def test_resolve_llm_runtime_llm_wait_fallback_mode_passthrough():
    orch = MagicMock()
    orch.config = {"llm": {"llm_wait_fallback_mode": "macro_cf30"}, "risk_management": {"params": {}}}
    rt = bridge.resolve_llm_runtime(orch)
    assert rt["llm_wait_fallback_mode"] == "macro_cf30"


def test_resolve_llm_runtime_normalizes_fields():
    orch = MagicMock()
    orch.config = {
        "llm": {
            "base_url": "http://x",
            "model": "m",
            "timeout_seconds": 7,
            "max_predict_tokens": 64,
            "keep_alive": "30m",
            "min_conviction_execute": 0.9,
            "ohlc_bars": 120,
            "analysis_granularity_seconds": 120,
            "analysis_bars": 120,
            "parse_retry_attempts": 3,
            "logic_line_max_chars": 180,
            "llm_extra_wall_clock_seconds": 0,
        },
        "risk_management": {"params": {"duration": 5, "duration_unit": "t"}},
    }
    rt = bridge.resolve_llm_runtime(orch)
    assert rt["base_url"] == "http://x"
    assert rt["model"] == "m"
    assert rt["timeout"] == 7.0
    assert rt["num_predict"] == 64
    assert rt["m3_bars"] >= 55
    assert rt["m15_bars"] >= 55
    assert rt["m5_bars"] >= 55
    assert rt["duration"] == 5
    assert rt["du"] == "t"
    assert rt["indicator_config"].hurst_window == 30


def test_resolve_llm_runtime_risk_limits_override_min_conviction():
    orch = MagicMock()
    orch.config = {
        "llm": {"min_conviction_execute": 0.5},
        "risk_management": {"params": {}, "limits": {"min_conviction_execute": 0.88}},
    }
    rt = bridge.resolve_llm_runtime(orch)
    assert rt["min_conviction_execute"] == pytest.approx(0.88)
