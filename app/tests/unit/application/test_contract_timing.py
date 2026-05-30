import pytest

from src.application.services.llm.contract_timing import (
    contract_duration_seconds,
    resolve_post_settlement_breath_seconds,
    resolve_refresh_entry_spacing_seconds,
)


def test_contract_duration_seconds_m1():
    assert contract_duration_seconds({"risk_management": {"params": {"duration": 1, "duration_unit": "m"}}}) == 60.0


def test_resolve_refresh_entry_spacing_m1_contract():
    cfg = {"risk_management": {"params": {"duration": 1, "duration_unit": "m"}}}
    orch = {
        "entry_spacing_follows_contract": True,
        "post_settlement_breath_seconds": 8,
        "cluster_refresh_entry_spacing_seconds": 5,
        "cluster_refresh_spacing_cap_seconds": 72,
    }
    spacing = resolve_refresh_entry_spacing_seconds(orch, cfg)
    assert spacing == 13.0


def test_resolve_post_settlement_breath_m1():
    cfg = {"risk_management": {"params": {"duration": 1, "duration_unit": "m"}}}
    breath = resolve_post_settlement_breath_seconds(
        {"post_settlement_breath_seconds": 8, "breath_follows_contract": True}, cfg
    )
    assert breath == 8.0


def test_contract_duration_seconds_other_units():
    assert contract_duration_seconds({"risk_management": {"params": {"duration": 30, "duration_unit": "s"}}}) == 30.0
    assert contract_duration_seconds({"risk_management": {"params": {"duration": 2, "duration_unit": "h"}}}) == 7200.0
    assert contract_duration_seconds({"risk_management": {"params": {"duration": "x", "duration_unit": "m"}}}) == 60.0


def test_resolve_refresh_entry_spacing_fixed_mode():
    spacing = resolve_refresh_entry_spacing_seconds(
        {
            "entry_spacing_follows_contract": False,
            "post_settlement_breath_seconds": 10,
            "cluster_refresh_entry_spacing_seconds": 4,
        },
        {},
    )
    assert spacing == 14.0


def test_resolve_post_settlement_breath_default_and_fixed():
    cfg = {"risk_management": {"params": {"duration": 1, "duration_unit": "m"}}}
    assert resolve_post_settlement_breath_seconds({"breath_follows_contract": True}, cfg) == pytest.approx(7.2)
    assert (
        resolve_post_settlement_breath_seconds(
            {"breath_follows_contract": False, "post_settlement_breath_seconds": 20}, cfg
        )
        == 20.0
    )
