from unittest.mock import MagicMock, patch

from src.application.services.llm.cluster_statarb_direction import direction_from_statarb_z
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


def test_apply_cluster_target_inverts_llm_side_on_divergence():
    orch = MagicMock()
    orch.config = {"llm": {"min_conviction_execute": 0.60}}
    orch._invert_quarantine_active = False
    orch._cluster_pause_after_loss_active = False
    decisions: dict = {}
    propagated, blocked, inverted, corrected = apply_cluster_target_decision(
        orch,
        target_sym="1HZ50V",
        target_direction=TradeDirection.CALL,
        index_note="STATARB_BEST leader=1HZ50V z=2.90",
        metrics={
            "conviction": 0.70,
            "macro_sentiment": "divergence_eu_leads",
            "macro_us_strength_quant": 0.40,
            "macro_eu_strength_quant": 0.80,
            "hmm_prob": 0.90,
            "hmm_state": 0,
            "index_m5_dir_by_symbol": {"1HZ50V": "down"},
            "statarb_spreads": {"1HZ50V": 2.9},
        },
        decisions=decisions,
        anchor_sym="R_100",
        conviction=0.70,
        macro_cfg={
            "assert_min_hmm_prob": 0.0,
            "allowed_execute_tags": ("divergence_eu_leads",),
            "statarb_z_threshold": 2.5,
            "confluence_conviction_floor": 0.60,
        },
        corr_cfg={
            "statarb_correct_llm_on_divergence": True,
            "quant_direction_stack_enabled": True,
            "cluster_invert_llm_side": True,
        },
        active_region="eu",
        exclusive=True,
        macro_tag="divergence_eu_leads",
        invert_on_block=False,
    )
    assert propagated == "1HZ50V[P]"
    assert blocked is None
    assert corrected is None
    assert inverted == "1HZ50V[C->P]"
    assert decisions["1HZ50V"]["direction"] == TradeDirection.PUT
    assert decisions["1HZ50V"]["metrics"]["execute"] is True
    assert decisions["1HZ50V"]["metrics"].get("llm_exec_inverted")


def test_apply_cluster_target_inverts_on_block_when_not_llm_side_invert():
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
            target_sym="R_50",
            target_direction=TradeDirection.CALL,
            index_note="STATARB_BEST leader=R_50 z=2.90",
            metrics={
                "conviction": 0.70,
                "macro_sentiment": "divergence_us_leads",
                "hmm_prob": 0.90,
                "hmm_state": 0,
            },
            decisions=decisions,
            anchor_sym="R_100",
            conviction=0.70,
            macro_cfg={"assert_min_hmm_prob": 0.0, "allowed_execute_tags": ("divergence_us_leads",)},
            corr_cfg={"cluster_invert_llm_side": False, "cluster_invert_on_block": True},
            active_region="us",
            exclusive=True,
            macro_tag="divergence_us_leads",
            invert_on_block=True,
        )
    assert propagated == "R_50[P]"
    assert inverted == "R_50[C->P]"
    assert decisions["R_50"]["metrics"]["llm_block_reason"] == "allowed_inverted"


def test_apply_cluster_target_inverts_on_divergence_statarb_block():
    orch = MagicMock()
    orch.config = {"llm": {"min_conviction_execute": 0.60}}
    orch._invert_quarantine_active = False
    orch._cluster_pause_after_loss_active = False
    decisions: dict = {}
    with patch(
        "src.application.services.llm.llm_cluster_target.cluster_execute_block_reason",
        return_value="allowed",
    ):
        propagated, blocked, inverted, corrected = apply_cluster_target_decision(
            orch,
            target_sym="R_50",
            target_direction=TradeDirection.CALL,
            index_note="STATARB_BEST leader=R_50 z=2.90",
            metrics={
                "conviction": 0.70,
                "macro_sentiment": "divergence_us_leads",
                "hmm_prob": 0.90,
                "hmm_state": 0,
            },
            decisions=decisions,
            anchor_sym="R_100",
            conviction=0.70,
            macro_cfg={"assert_min_hmm_prob": 0.0, "allowed_execute_tags": ("divergence_us_leads",)},
            corr_cfg={"cluster_invert_llm_side": True, "cluster_invert_on_block": True},
            active_region="us",
            exclusive=True,
            macro_tag="divergence_us_leads",
            invert_on_block=True,
        )
    assert propagated == "R_50[P]"
    assert inverted == "R_50[C->P]"
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
            target_sym="R_50",
            target_direction=TradeDirection.CALL,
            index_note="STATARB_BEST",
            metrics={
                "conviction": 0.70,
                "macro_sentiment": "divergence_us_leads",
                "hmm_prob": 0.90,
                "hmm_state": 0,
            },
            decisions=decisions,
            anchor_sym="R_100",
            conviction=0.70,
            macro_cfg={"assert_min_hmm_prob": 0.0, "allowed_execute_tags": ("divergence_us_leads",)},
            corr_cfg={"cluster_invert_llm_side": True, "cluster_invert_on_block": True},
            active_region="eu",
            exclusive=True,
            macro_tag="divergence_us_leads",
            invert_on_block=True,
        )
    assert blocked is not None
    assert decisions["R_50"]["direction"] == TradeDirection.PUT


def test_apply_cluster_target_keeps_eu_cluster_llm_on_divergence():
    orch = MagicMock()
    orch.config = {"llm": {"min_conviction_execute": 0.60}}
    orch._invert_quarantine_active = False
    orch._cluster_pause_after_loss_active = False
    decisions: dict = {}
    apply_cluster_target_decision(
        orch,
        target_sym="R_75",
        target_direction=TradeDirection.CALL,
        index_note="M5",
        metrics={
            "conviction": 0.70,
            "macro_sentiment": "divergence_eu_leads",
            "eu_cluster": "CALL",
            "index_m5_dir_by_symbol": {"R_75": "down"},
            "statarb_spreads": {"R_75": 0.2},
            "hmm_state": 0,
            "hmm_prob": 0.90,
        },
        decisions=decisions,
        anchor_sym="R_100",
        conviction=0.70,
        macro_cfg={
            "assert_min_hmm_prob": 0.0,
            "allowed_execute_tags": ("divergence_eu_leads",),
            "statarb_z_threshold": 2.5,
        },
        corr_cfg={"quant_direction_stack_enabled": True, "cluster_invert_llm_side": False},
        active_region="eu",
        exclusive=True,
        macro_tag="divergence_eu_leads",
        invert_on_block=False,
    )
    assert decisions["R_75"]["direction"] == TradeDirection.CALL
    assert decisions["R_75"]["metrics"].get("eu_cluster") != "PUT"
