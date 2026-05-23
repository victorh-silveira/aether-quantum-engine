from src.application.services.llm.llm_refresh_policy import (
    SCHEDULE_ALWAYS,
    SCHEDULE_DAILY,
    SCHEDULE_TAG_CHANGE,
    macro_tag_allows_llm_call,
    resolve_llm_refresh_schedule,
    should_refresh_llm_decision,
)
from src.application.services.llm.macro_config import MacroSnapshot, resolve_macro_config


def _snap(tag: str) -> MacroSnapshot:
    return MacroSnapshot(
        us_dir="up",
        eu_dir="up",
        us_strength=0.9,
        eu_strength=0.88,
        tag=tag,
        eurusd_bias="CALL",
        cluster_status="",
        macro_block="",
        fx_reference_line="",
        us_parts=(),
        eu_parts=(),
    )


def test_resolve_llm_refresh_schedule_default():
    assert resolve_llm_refresh_schedule({}) == SCHEDULE_TAG_CHANGE
    assert resolve_llm_refresh_schedule({"llm": {"refresh_schedule": "always"}}) == "always"
    assert resolve_llm_refresh_schedule({"llm": {"refresh_schedule": "invalid"}}) == SCHEDULE_TAG_CHANGE


def test_should_refresh_always_and_daily():
    assert should_refresh_llm_decision(
        schedule=SCHEDULE_ALWAYS,
        current_tag="risk_off",
        last_tag="risk_off",
        has_cached_decisions=True,
    )
    assert should_refresh_llm_decision(
        schedule=SCHEDULE_DAILY,
        current_tag="risk_off",
        last_tag="risk_off",
        has_cached_decisions=True,
    )


def test_macro_tag_allows_when_tag_permitted():
    ok, note = macro_tag_allows_llm_call(
        _snap("risk_off"),
        {"allowed_execute_tags": ["risk_off"]},
    )
    assert ok is True
    assert note == ""


def test_should_refresh_on_tag_change():
    assert (
        should_refresh_llm_decision(
            schedule=SCHEDULE_TAG_CHANGE,
            current_tag="risk_off",
            last_tag="risk_off",
            has_cached_decisions=True,
        )
        is False
    )
    assert (
        should_refresh_llm_decision(
            schedule=SCHEDULE_TAG_CHANGE,
            current_tag="risk_off",
            last_tag="risk_on",
            has_cached_decisions=True,
        )
        is True
    )
    assert (
        should_refresh_llm_decision(
            schedule=SCHEDULE_TAG_CHANGE,
            current_tag="risk_off",
            last_tag=None,
            has_cached_decisions=False,
        )
        is True
    )


def test_macro_tag_allows_llm_call():
    ok, _ = macro_tag_allows_llm_call(
        _snap("risk_off"),
        {"allowed_execute_tags": ["risk_off", "divergence_us_leads"]},
    )
    assert ok is True
    ok2, note = macro_tag_allows_llm_call(
        _snap("risk_on"),
        {"allowed_execute_tags": ["risk_off"]},
    )
    assert ok2 is False
    assert "risk_on" in note


def test_resolve_macro_config_empty_allowed_tags():
    cfg = resolve_macro_config({"allowed_execute_tags": []})
    assert cfg.get("allowed_execute_tags") is None


def test_resolve_macro_config_invalid_allowed_tags_type():
    cfg = resolve_macro_config({"allowed_execute_tags": "risk_off"})
    assert cfg.get("allowed_execute_tags") is None
