import pytest

from src.application.services.llm.llm_config_merge import (
    effective_llm_section,
    merge_execution_section,
    risk_limits_section,
)


def test_effective_llm_section_llm_config_wins_last():
    root = {
        "llm": {"model": "x"},
        "gemini": {"model": "gflash", "generation_config": {"top_p": 0.9}},
        "llm_config": {"model": "final", "system_prompt": "ctx"},
    }
    eff = effective_llm_section(root)
    assert eff["model"] == "final"
    assert eff["system_prompt"] == "ctx"
    assert eff["generation_config"]["top_p"] == pytest.approx(0.9)


def test_effective_llm_gemini_merges_temperature_into_generation_config():
    root = {"llm": {}, "gemini": {"temperature": 0.2}}
    eff = effective_llm_section(root)
    assert eff["generation_config"]["temperature"] == pytest.approx(0.2)


def test_effective_llm_gemini_overrides_model_before_llm_config():
    root = {"llm": {"model": "base"}, "gemini": {"model": "gemini-2.5-flash"}, "llm_config": {}}
    eff = effective_llm_section(root)
    assert eff["model"] == "gemini-2.5-flash"


def test_effective_llm_llm_config_generation_config_merges_over_gemini():
    root = {
        "llm": {},
        "gemini": {"generation_config": {"temperature": 0.1, "max_output_tokens": 10}},
        "llm_config": {"generation_config": {"temperature": 0.9}},
    }
    eff = effective_llm_section(root)
    assert eff["generation_config"]["temperature"] == pytest.approx(0.9)
    assert eff["generation_config"]["max_output_tokens"] == 10


def test_effective_llm_gemini_timeout_system_prompt_num_predict():
    root = {
        "llm": {},
        "gemini": {"timeout_seconds": 15, "system_prompt": "extra", "num_predict": 48},
    }
    eff = effective_llm_section(root)
    assert eff["timeout_seconds"] == pytest.approx(15.0)
    assert eff["system_prompt"] == "extra"
    assert eff["max_predict_tokens"] == 48


def test_effective_llm_user_llm_wins_timeout_tokens_over_gemini():
    root = {
        "llm": {"timeout_seconds": 18, "max_predict_tokens": 32},
        "gemini": {"timeout_seconds": 28, "num_predict": 64},
    }
    eff = effective_llm_section(root)
    assert eff["timeout_seconds"] == pytest.approx(18.0)
    assert eff["max_predict_tokens"] == 32


def test_effective_llm_user_generation_config_wins_over_gemini():
    root = {
        "llm": {"generation_config": {"max_output_tokens": 48}},
        "gemini": {"generation_config": {"max_output_tokens": 10, "temperature": 0.2}},
    }
    eff = effective_llm_section(root)
    assert eff["generation_config"]["max_output_tokens"] == 48
    assert eff["generation_config"]["temperature"] == pytest.approx(0.2)


def test_effective_llm_user_system_prompt_wins_over_gemini():
    root = {"llm": {"system_prompt": "user_ctx"}, "gemini": {"system_prompt": "gem_ctx"}}
    eff = effective_llm_section(root)
    assert eff["system_prompt"] == "user_ctx"


def test_merge_execution_section_skips_when_execution_absent():
    cfg = {"orchestrator": {"cycle_interval_seconds": 9}}
    merge_execution_section(cfg)
    assert cfg["orchestrator"]["cycle_interval_seconds"] == 9


def test_merge_execution_section_writes_orchestrator():
    cfg = {"orchestrator": {"cycle_interval_seconds": 1, "execution": {"settlement_poll_seconds": 9.0}}}
    cfg["execution"] = {
        "cycle_interval_seconds": 7,
        "settlement_poll_seconds": 2.0,
        "include_anchor_trades": False,
    }
    merge_execution_section(cfg)
    assert cfg["orchestrator"]["cycle_interval_seconds"] == 7
    assert cfg["orchestrator"]["execution"]["settlement_poll_seconds"] == 2.0
    assert cfg["orchestrator"]["execution"]["include_anchor_trades"] is False


def test_merge_execution_section_creates_orchestrator():
    cfg: dict = {"execution": {"cycle_interval_seconds": 3, "settlement_poll_seconds": 1.1}}
    merge_execution_section(cfg)
    assert cfg["orchestrator"]["cycle_interval_seconds"] == 3
    assert cfg["orchestrator"]["execution"]["settlement_poll_seconds"] == 1.1


def test_risk_limits_section_empty():
    assert risk_limits_section({}) == {}
    assert risk_limits_section({"risk_management": {}}) == {}


def test_risk_limits_section_reads_limits():
    root = {"risk_management": {"limits": {"min_conviction_execute": 0.7, "stop_loss_pct": 5.0}}}
    lim = risk_limits_section(root)
    assert lim["min_conviction_execute"] == 0.7
    assert lim["stop_loss_pct"] == 5.0


def test_risk_limits_rm_not_dict():
    assert risk_limits_section({"risk_management": None}) == {}


def test_effective_llm_lc_full_branch():
    root = {
        "llm": {"options": {}},
        "llm_config": {
            "model": "m2",
            "timeout_seconds": 3,
            "keep_alive": "1m",
            "system_prompt": "s",
            "num_predict": "x",
            "temperature": "y",
            "base_url": "http://z",
        },
    }
    eff = effective_llm_section(root)
    assert eff["model"] == "m2"
    assert eff["timeout_seconds"] == 3.0
    assert eff["base_url"] == "http://z"


def test_merge_execution_nested_execution_subdict():
    cfg = {"execution": {"cycle_interval_seconds": 1, "execution": {"settlement_poll_seconds": 8.8}}}
    merge_execution_section(cfg)
    assert cfg["orchestrator"]["execution"]["settlement_poll_seconds"] == pytest.approx(8.8)
