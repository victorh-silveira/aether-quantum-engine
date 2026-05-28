from unittest.mock import MagicMock

from src.application.services.llm.llm_cluster_target import apply_cluster_target_decision
from src.domain.models.trade import TradeDirection


def _base_metrics(**overrides):
    base = {
        "conviction": 0.70,
        "execute": False,
        "us_cluster": "PUT",
        "eu_cluster": "PUT",
        "macro_sentiment": "risk_off",
        "macro_us_strength_quant": 0.30,
        "macro_eu_strength_quant": 0.72,
        "hmm_prob": 0.90,
    }
    base.update(overrides)
    return base


def test_apply_cluster_target_does_not_invert_outside_divergence_tag():
    orch = MagicMock()
    orch.config = {"llm": {"min_conviction_execute": 0.60}}
    orch._invert_quarantine_active = False
    decisions: dict = {}
    propagated, blocked, inverted = apply_cluster_target_decision(
        orch,
        target_sym="OTC_FTSE",
        target_direction=TradeDirection.PUT,
        index_note="STATARB_NO_Z_ALIGN",
        metrics=_base_metrics(conviction=0.70, statarb_spreads={"OTC_FTSE": -2.0}),
        decisions=decisions,
        anchor_sym="frxEURUSD",
        conviction=0.70,
        macro_cfg={"assert_min_hmm_prob": 0.0, "allowed_execute_tags": ("risk_off",)},
        corr_cfg={"statarb_require_z_align": True, "cluster_invert_on_block": True},
        active_region="eu",
        exclusive=True,
        macro_tag="risk_off",
        invert_on_block=True,
    )
    assert propagated is None
    assert blocked == "OTC_FTSE[P]:statarb_z_misaligned"
    assert inverted is None
    assert decisions["OTC_FTSE"]["direction"] == TradeDirection.PUT
    assert decisions["OTC_FTSE"]["metrics"]["execute"] is False
    assert decisions["OTC_FTSE"]["metrics"]["llm_block_reason"] == "statarb_z_misaligned"


def test_apply_cluster_target_inverts_on_divergence_tag():
    orch = MagicMock()
    orch.config = {"llm": {"min_conviction_execute": 0.60}}
    orch._invert_quarantine_active = False
    decisions: dict = {}
    propagated, blocked, inverted = apply_cluster_target_decision(
        orch,
        target_sym="OTC_FTSE",
        target_direction=TradeDirection.PUT,
        index_note="STATARB_NO_Z_ALIGN",
        metrics=_base_metrics(
            conviction=0.70,
            macro_sentiment="divergence_eu_leads",
            statarb_spreads={"OTC_FTSE": -2.0},
        ),
        decisions=decisions,
        anchor_sym="frxEURUSD",
        conviction=0.70,
        macro_cfg={"assert_min_hmm_prob": 0.0, "allowed_execute_tags": ("divergence_eu_leads",)},
        corr_cfg={"statarb_require_z_align": True, "cluster_invert_on_block": True},
        active_region="eu",
        exclusive=True,
        macro_tag="divergence_eu_leads",
        invert_on_block=True,
    )
    assert propagated == "OTC_FTSE[C]"
    assert blocked is None
    assert inverted == "OTC_FTSE[P->C]"
    assert decisions["OTC_FTSE"]["direction"] == TradeDirection.CALL
    assert decisions["OTC_FTSE"]["metrics"]["execute"] is True
    assert decisions["OTC_FTSE"]["metrics"]["llm_exec_inverted"] is True


def test_apply_cluster_target_respects_quarantine_after_loss():
    orch = MagicMock()
    orch.config = {"llm": {"min_conviction_execute": 0.60}}
    orch._invert_quarantine_active = True
    decisions: dict = {}
    propagated, blocked, inverted = apply_cluster_target_decision(
        orch,
        target_sym="OTC_FTSE",
        target_direction=TradeDirection.PUT,
        index_note="STATARB_NO_Z_ALIGN",
        metrics=_base_metrics(
            conviction=0.70,
            macro_sentiment="divergence_eu_leads",
            statarb_spreads={"OTC_FTSE": -2.0},
        ),
        decisions=decisions,
        anchor_sym="frxEURUSD",
        conviction=0.70,
        macro_cfg={"assert_min_hmm_prob": 0.0, "allowed_execute_tags": ("divergence_eu_leads",)},
        corr_cfg={"statarb_require_z_align": True, "cluster_invert_on_block": True},
        active_region="eu",
        exclusive=True,
        macro_tag="divergence_eu_leads",
        invert_on_block=True,
    )
    assert propagated is None
    assert blocked == "OTC_FTSE[P]:invert_quarantine_after_loss"
    assert inverted is None
    assert decisions["OTC_FTSE"]["direction"] == TradeDirection.PUT
    assert decisions["OTC_FTSE"]["metrics"]["execute"] is False
    assert decisions["OTC_FTSE"]["metrics"]["llm_block_reason"] == "invert_quarantine_after_loss"
