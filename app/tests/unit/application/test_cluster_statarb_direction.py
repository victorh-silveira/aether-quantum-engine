from unittest.mock import MagicMock, patch

from src.application.services.llm.cluster_statarb_direction import (
    correct_cluster_direction_for_divergence,
    direction_from_statarb_z,
)
from src.application.services.llm.llm_cluster_target import apply_cluster_target_decision
from src.domain.models.trade import TradeDirection


def test_direction_from_statarb_z_mean_reversion():
    assert direction_from_statarb_z(2.8, hmm_state=0, z_threshold=2.5, min_abs_z=0.65) == TradeDirection.PUT
    assert direction_from_statarb_z(-2.8, hmm_state=0, z_threshold=2.5, min_abs_z=0.65) == TradeDirection.CALL
    assert direction_from_statarb_z(1.0, hmm_state=0, z_threshold=2.5, min_abs_z=0.65) is None
    assert direction_from_statarb_z(0.2, hmm_state=0, z_threshold=2.5, min_abs_z=0.65) is None


def test_direction_from_statarb_z_trending():
    assert direction_from_statarb_z(0.70, hmm_state=1, z_threshold=2.5, min_abs_z=0.65) == TradeDirection.CALL
    assert direction_from_statarb_z(-0.70, hmm_state=1, z_threshold=2.5, min_abs_z=0.65) == TradeDirection.PUT


def test_correct_cluster_direction_call_to_put_on_positive_z():
    direction, corrected, note = correct_cluster_direction_for_divergence(
        TradeDirection.CALL,
        macro_tag="divergence_us_leads",
        target_sym="OTC_DJI",
        metrics={"statarb_spreads": {"OTC_DJI": 2.9}, "hmm_state": 0},
        corr_cfg={"statarb_correct_llm_on_divergence": True},
        macro_cfg={"statarb_z_threshold": 2.5, "statarb_min_abs_z_by_tag": {"divergence_us_leads": 0.65}},
    )
    assert corrected is True
    assert direction == TradeDirection.PUT
    assert "STATARB_DIR" in note


def test_correct_cluster_direction_m5_overrides_llm_call():
    direction, corrected, note = correct_cluster_direction_for_divergence(
        TradeDirection.CALL,
        macro_tag="divergence_us_leads",
        target_sym="OTC_DJI",
        metrics={
            "index_m5_dir_by_symbol": {"OTC_DJI": "down"},
            "statarb_spreads": {"OTC_DJI": 0.2},
            "hmm_state": 0,
        },
        corr_cfg={"statarb_correct_llm_on_divergence": True},
        macro_cfg={"statarb_z_threshold": 2.5},
    )
    assert corrected is True
    assert direction == TradeDirection.PUT
    assert "M5_DIR" in note


def test_correct_cluster_direction_ignores_flat_m5_micro():
    direction, corrected, note = correct_cluster_direction_for_divergence(
        TradeDirection.CALL,
        macro_tag="divergence_us_leads",
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


def test_correct_cluster_direction_keeps_when_z_supports_direction():
    direction, corrected, note = correct_cluster_direction_for_divergence(
        TradeDirection.PUT,
        macro_tag="divergence_us_leads",
        target_sym="OTC_DJI",
        metrics={"statarb_spreads": {"OTC_DJI": 2.9}, "hmm_state": 0},
        corr_cfg={"quant_direction_stack_enabled": True},
        macro_cfg={"statarb_z_threshold": 2.5, "statarb_min_abs_z_by_tag": {"divergence_us_leads": 0.65}},
    )
    assert corrected is False
    assert direction == TradeDirection.PUT
    assert note == ""


def test_apply_cluster_target_corrects_llm_call_to_put_on_divergence():
    orch = MagicMock()
    orch.config = {"llm": {"min_conviction_execute": 0.60}}
    orch._invert_quarantine_active = False
    orch._cluster_pause_after_loss_active = False
    decisions: dict = {}
    propagated, blocked, inverted, corrected = apply_cluster_target_decision(
        orch,
        target_sym="OTC_DJI",
        target_direction=TradeDirection.CALL,
        index_note="STATARB_BEST leader=OTC_DJI z=2.90",
        metrics={
            "conviction": 0.70,
            "macro_sentiment": "divergence_us_leads",
            "macro_us_strength_quant": 0.80,
            "macro_eu_strength_quant": 0.40,
            "hmm_prob": 0.90,
            "hmm_state": 0,
            "index_m5_dir_by_symbol": {"OTC_DJI": "down"},
            "statarb_spreads": {"OTC_DJI": 0.2},
        },
        decisions=decisions,
        anchor_sym="frxEURUSD",
        conviction=0.70,
        macro_cfg={
            "assert_min_hmm_prob": 0.0,
            "allowed_execute_tags": ("divergence_us_leads",),
            "statarb_z_threshold": 2.5,
            "statarb_min_abs_z_by_tag": {"divergence_us_leads": 0.65},
            "confluence_conviction_floor": 0.65,
        },
        corr_cfg={"statarb_correct_llm_on_divergence": True},
        active_region="us",
        exclusive=True,
        macro_tag="divergence_us_leads",
        invert_on_block=False,
    )
    assert propagated == "OTC_DJI[P]"
    assert blocked is None
    assert corrected == "OTC_DJI[C->P]"
    assert inverted is None
    assert decisions["OTC_DJI"]["direction"] == TradeDirection.PUT
    assert decisions["OTC_DJI"]["metrics"]["execute"] is True
    assert decisions["OTC_DJI"]["metrics"]["llm_statarb_dir_corrected"] is True


def test_apply_cluster_target_inverts_on_divergence_statarb_block():
    orch = MagicMock()
    orch.config = {"llm": {"min_conviction_execute": 0.60}}
    orch._invert_quarantine_active = False
    orch._cluster_pause_after_loss_active = False
    decisions: dict = {}
    with patch(
        "src.application.services.llm.llm_cluster_target.cluster_execute_block_reason",
        return_value="statarb_z_misaligned",
    ):
        propagated, blocked, inverted, corrected = apply_cluster_target_decision(
            orch,
            target_sym="OTC_DJI",
            target_direction=TradeDirection.CALL,
            index_note="STATARB_BEST leader=OTC_DJI z=2.90",
            metrics={
                "conviction": 0.70,
                "macro_sentiment": "divergence_us_leads",
                "hmm_prob": 0.90,
                "hmm_state": 0,
            },
            decisions=decisions,
            anchor_sym="frxEURUSD",
            conviction=0.70,
            macro_cfg={"assert_min_hmm_prob": 0.0, "allowed_execute_tags": ("divergence_us_leads",)},
            corr_cfg={"cluster_invert_on_block": True},
            active_region="us",
            exclusive=True,
            macro_tag="divergence_us_leads",
            invert_on_block=True,
        )
    assert propagated == "OTC_DJI[P]"
    assert inverted == "OTC_DJI[C->P]"
    assert corrected is None


def test_apply_cluster_target_quarantine_sets_invert_block_reason():
    orch = MagicMock()
    orch.config = {"llm": {"min_conviction_execute": 0.60}}
    orch._invert_quarantine_active = True
    orch._cluster_pause_after_loss_active = False
    decisions: dict = {}
    with patch(
        "src.application.services.llm.llm_cluster_target.cluster_execute_block_reason",
        return_value="statarb_z_misaligned",
    ):
        _, blocked, _, _ = apply_cluster_target_decision(
            orch,
            target_sym="OTC_DJI",
            target_direction=TradeDirection.CALL,
            index_note="STATARB_BEST",
            metrics={
                "conviction": 0.70,
                "macro_sentiment": "divergence_us_leads",
                "hmm_prob": 0.90,
                "hmm_state": 0,
            },
            decisions=decisions,
            anchor_sym="frxEURUSD",
            conviction=0.70,
            macro_cfg={"assert_min_hmm_prob": 0.0, "allowed_execute_tags": ("divergence_us_leads",)},
            corr_cfg={"cluster_invert_on_block": True},
            active_region="eu",
            exclusive=True,
            macro_tag="divergence_us_leads",
            invert_on_block=True,
        )
    assert blocked is not None
    assert decisions["OTC_DJI"]["metrics"]["llm_block_reason"] == "invert_quarantine_after_loss"


def test_apply_cluster_target_updates_eu_cluster_on_correct():
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
            "macro_sentiment": "divergence_eu_leads",
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
            "allowed_execute_tags": ("divergence_eu_leads",),
            "statarb_z_threshold": 2.5,
        },
        corr_cfg={"quant_direction_stack_enabled": True},
        active_region="eu",
        exclusive=True,
        macro_tag="divergence_eu_leads",
        invert_on_block=False,
    )
    assert decisions["OTC_FCHI"]["metrics"]["eu_cluster"] == "PUT"
