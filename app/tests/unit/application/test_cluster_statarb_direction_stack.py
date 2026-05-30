from unittest.mock import MagicMock

from src.application.services.llm.cluster_statarb_direction import (
    correct_cluster_direction_for_divergence,
    correct_cluster_direction_for_tag,
    quant_direction_stack_enabled,
)
from src.application.services.llm.llm_cluster_target import apply_cluster_target_decision
from src.domain.models.trade import TradeDirection


def test_quant_direction_stack_enabled_reads_corr_flag():
    assert quant_direction_stack_enabled({"quant_direction_stack_enabled": False}) is False
    assert quant_direction_stack_enabled({"statarb_correct_llm_on_divergence": True}) is True


def test_correct_cluster_direction_disabled_stack():
    direction, corrected, note = correct_cluster_direction_for_tag(
        TradeDirection.CALL,
        macro_tag="risk_on",
        target_sym="OTC_DJI",
        metrics={"statarb_spreads": {"OTC_DJI": 2.9}, "hmm_state": 0},
        corr_cfg={"quant_direction_stack_enabled": False},
        macro_cfg={},
    )
    assert corrected is False
    assert direction == TradeDirection.CALL
    assert note == ""


def test_correct_cluster_direction_no_statarb_spreads():
    direction, corrected, note = correct_cluster_direction_for_tag(
        TradeDirection.CALL,
        macro_tag="risk_on",
        target_sym="OTC_DJI",
        metrics={"hmm_state": 0},
        corr_cfg={"quant_direction_stack_enabled": True},
        macro_cfg={},
    )
    assert corrected is False
    assert note == ""


def test_correct_cluster_direction_keeps_when_z_supports_direction():
    direction, corrected, note = correct_cluster_direction_for_tag(
        TradeDirection.PUT,
        macro_tag="risk_on",
        target_sym="OTC_DJI",
        metrics={"statarb_spreads": {"OTC_DJI": 2.9}, "hmm_state": 0},
        corr_cfg={"quant_direction_stack_enabled": True},
        macro_cfg={"statarb_z_threshold": 2.5, "statarb_min_abs_z_by_tag": {"risk_on": 0.65}},
    )
    assert corrected is False
    assert direction == TradeDirection.PUT
    assert note == ""


def test_correct_cluster_direction_keeps_llm_on_divergence_tags():
    direction, corrected, note = correct_cluster_direction_for_divergence(
        TradeDirection.CALL,
        macro_tag="divergence_eu_leads",
        target_sym="1HZ50V",
        metrics={
            "index_m5_dir_by_symbol": {"1HZ50V": "down"},
            "statarb_spreads": {"1HZ50V": 2.9},
            "hmm_state": 0,
        },
        corr_cfg={"statarb_correct_llm_on_divergence": True, "quant_direction_stack_enabled": True},
        macro_cfg={"statarb_z_threshold": 2.5},
    )
    assert corrected is False
    assert direction == TradeDirection.CALL
    assert note == ""


def test_correct_cluster_direction_keeps_when_statarb_implied_matches_llm():
    direction, corrected, note = correct_cluster_direction_for_tag(
        TradeDirection.CALL,
        macro_tag="risk_on",
        target_sym="OTC_DJI",
        metrics={"statarb_spreads": {"OTC_DJI": -2.8}, "hmm_state": 0},
        corr_cfg={"quant_direction_stack_enabled": True},
        macro_cfg={"statarb_z_threshold": 2.5, "statarb_min_abs_z_by_tag": {"risk_on": 0.65}},
    )
    assert corrected is False
    assert direction == TradeDirection.CALL


def test_correct_cluster_direction_statarb_on_risk_on():
    direction, corrected, note = correct_cluster_direction_for_tag(
        TradeDirection.CALL,
        macro_tag="risk_on",
        target_sym="OTC_DJI",
        metrics={"statarb_spreads": {"OTC_DJI": 2.9}, "hmm_state": 0},
        corr_cfg={"quant_direction_stack_enabled": True},
        macro_cfg={"statarb_z_threshold": 2.5, "statarb_min_abs_z_by_tag": {"risk_on": 0.65}},
    )
    assert corrected is True
    assert direction == TradeDirection.PUT
    assert "STATARB_DIR" in note


def test_correct_cluster_direction_m5_overrides_llm_on_risk_on():
    direction, corrected, note = correct_cluster_direction_for_tag(
        TradeDirection.CALL,
        macro_tag="risk_on",
        target_sym="OTC_DJI",
        metrics={
            "index_m5_dir_by_symbol": {"OTC_DJI": "down"},
            "statarb_spreads": {"OTC_DJI": 0.2},
            "hmm_state": 0,
        },
        corr_cfg={"quant_direction_stack_enabled": True},
        macro_cfg={"statarb_z_threshold": 2.5},
    )
    assert corrected is True
    assert direction == TradeDirection.PUT
    assert "M5_DIR" in note


def test_correct_cluster_direction_ignores_flat_m5_micro():
    direction, corrected, note = correct_cluster_direction_for_tag(
        TradeDirection.CALL,
        macro_tag="risk_on",
        target_sym="OTC_DJI",
        metrics={
            "index_m5_dir_by_symbol": {"OTC_DJI": "flat"},
            "statarb_spreads": {"OTC_DJI": 0.2},
            "hmm_state": 0,
        },
        corr_cfg={"quant_direction_stack_enabled": True},
        macro_cfg={"statarb_z_threshold": 2.5},
    )
    assert corrected is False
    assert direction == TradeDirection.CALL
    assert note == ""


def test_apply_cluster_target_updates_us_cluster_on_risk_on_correct():
    orch = MagicMock()
    orch.config = {"llm": {"min_conviction_execute": 0.60}}
    orch._invert_quarantine_active = False
    orch._cluster_pause_after_loss_active = False
    decisions: dict = {}
    apply_cluster_target_decision(
        orch,
        target_sym="OTC_DJI",
        target_direction=TradeDirection.CALL,
        index_note="M5",
        metrics={
            "conviction": 0.70,
            "macro_sentiment": "risk_on",
            "us_cluster": "CALL",
            "index_m5_dir_by_symbol": {"OTC_DJI": "down"},
            "statarb_spreads": {"OTC_DJI": 0.2},
            "hmm_state": 0,
            "hmm_prob": 0.90,
        },
        decisions=decisions,
        anchor_sym="frxEURUSD",
        conviction=0.70,
        macro_cfg={
            "assert_min_hmm_prob": 0.0,
            "allowed_execute_tags": ("risk_on",),
            "statarb_z_threshold": 2.5,
            "confluence_conviction_floor": 0.60,
        },
        corr_cfg={"quant_direction_stack_enabled": True},
        active_region="us",
        exclusive=True,
        macro_tag="risk_on",
        invert_on_block=False,
    )
    assert decisions["OTC_DJI"]["metrics"]["us_cluster"] == "PUT"


def test_apply_cluster_target_updates_eu_cluster_on_risk_off_correct():
    orch = MagicMock()
    orch.config = {"llm": {"min_conviction_execute": 0.60}}
    orch._invert_quarantine_active = False
    orch._cluster_pause_after_loss_active = False
    decisions: dict = {}
    apply_cluster_target_decision(
        orch,
        target_sym="OTC_FCHI",
        target_direction=TradeDirection.CALL,
        index_note="M5",
        metrics={
            "conviction": 0.70,
            "macro_sentiment": "risk_off",
            "eu_cluster": "CALL",
            "index_m5_dir_by_symbol": {"OTC_FCHI": "down"},
            "statarb_spreads": {"OTC_FCHI": 0.2},
            "hmm_state": 0,
            "hmm_prob": 0.90,
        },
        decisions=decisions,
        anchor_sym="frxEURUSD",
        conviction=0.70,
        macro_cfg={
            "assert_min_hmm_prob": 0.0,
            "allowed_execute_tags": ("risk_off",),
            "confluence_conviction_floor": 0.60,
        },
        corr_cfg={"quant_direction_stack_enabled": True},
        active_region="eu",
        exclusive=True,
        macro_tag="risk_off",
        invert_on_block=False,
    )
    assert decisions["OTC_FCHI"]["metrics"]["eu_cluster"] == "PUT"
