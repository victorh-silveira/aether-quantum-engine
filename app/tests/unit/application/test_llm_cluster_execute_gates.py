from unittest.mock import MagicMock

from src.application.services.llm.llm_cluster_guards import (
    cluster_execute_block_reason,
    cluster_execute_flag,
)
from src.application.services.llm.llm_cluster_propagate import propagate_cluster_decisions
from src.domain.models.trade import TradeDirection
from tests.unit.application.cluster_guard_metrics import base_cluster_metrics


def test_cluster_pause_after_loss_blocks_execute():
    orch = MagicMock()
    orch.config = {"llm": {"min_conviction_execute": 0.60}}
    orch._cluster_pause_after_loss_active = True
    macro = {"confluence_conviction_floor": 0.65, "assert_min_hmm_prob": 0.0}
    corr = {"statarb_require_z_align": True}
    metrics = base_cluster_metrics(statarb_spreads={"OTC_DJI": -1.5}, hmm_state=0)
    assert (
        cluster_execute_block_reason(
            orch,
            metrics,
            0.70,
            TradeDirection.CALL,
            macro,
            corr,
            active_region="us",
            target_sym="OTC_DJI",
            llm_cluster_explicit=True,
            index_note="STATARB_BEST leader=OTC_DJI z=-1.50",
        )
        == "cluster_pause_after_loss"
    )


def test_cluster_execute_statarb_best_note_uses_tag_min_abs_at_gate():
    orch = MagicMock()
    orch.config = {"llm": {"min_conviction_execute": 0.60}}
    macro = {
        "confluence_conviction_floor": 0.65,
        "assert_min_hmm_prob": 0.0,
        "statarb_min_abs_z_by_tag": {"risk_on": 0.50},
    }
    corr = {"statarb_require_z_align": True, "statarb_index_min_abs_z": 0.85}
    metrics = base_cluster_metrics(
        macro_sentiment="risk_on",
        macro_us_strength_quant=0.80,
        macro_eu_strength_quant=0.30,
        statarb_spreads={"OTC_DJI": -0.55},
        hmm_state=0,
    )
    assert (
        cluster_execute_block_reason(
            orch,
            metrics,
            0.72,
            TradeDirection.CALL,
            macro,
            corr,
            active_region="us",
            target_sym="OTC_DJI",
            llm_cluster_explicit=True,
            index_note="STATARB_BEST leader=OTC_DJI z=-0.55 score=0.55",
        )
        == "allowed"
    )


def test_cluster_execute_weak_note_still_requires_z_align():
    orch = MagicMock()
    orch.config = {"llm": {"min_conviction_execute": 0.60}}
    macro = {
        "confluence_conviction_floor": 0.65,
        "assert_min_hmm_prob": 0.0,
        "statarb_min_abs_z_by_tag": {"risk_on": 0.65},
    }
    corr = {"statarb_require_z_align": True, "statarb_index_min_abs_z": 0.85}
    metrics = base_cluster_metrics(
        macro_sentiment="risk_on",
        macro_us_strength_quant=0.80,
        statarb_spreads={"OTC_SPC": -0.35},
        hmm_state=0,
    )
    assert (
        cluster_execute_block_reason(
            orch,
            metrics,
            0.70,
            TradeDirection.CALL,
            macro,
            corr,
            active_region="us",
            target_sym="OTC_SPC",
            llm_cluster_explicit=True,
            index_note="STATARB_WEAK leader=OTC_SPC z=-0.35",
        )
        == "statarb_z_misaligned"
    )
    assert (
        cluster_execute_block_reason(
            orch,
            metrics,
            0.70,
            TradeDirection.CALL,
            macro,
            corr,
            active_region="us",
            target_sym="OTC_SPC",
            llm_cluster_explicit=True,
            index_note="",
        )
        == "statarb_z_misaligned"
    )


def test_cluster_execute_flag_conviction_and_direction_gates():
    orch = MagicMock()
    orch.config = {"llm": {"min_conviction_execute": 0.60}}
    macro = {"confluence_conviction_floor": 0.65, "assert_min_hmm_prob": 0.0}
    corr = {"statarb_require_z_align": True}
    assert (
        cluster_execute_flag(
            orch,
            base_cluster_metrics(),
            0.50,
            TradeDirection.PUT,
            macro,
            corr,
            active_region="eu",
            target_sym="OTC_FCHI",
        )
        is False
    )
    assert (
        cluster_execute_flag(
            orch, base_cluster_metrics(), 0.70, None, macro, corr, active_region="eu", target_sym="OTC_FCHI"
        )
        is False
    )


def test_cluster_propagate_logs_empty_when_cluster_tags_missing():
    orch = MagicMock()
    orch.anchor = "frxEURUSD"
    orch.symbols = ["frxEURUSD", "OTC_FCHI"]
    orch.config = {
        "llm": {"min_conviction_execute": 0.60},
        "strategy": {
            "clusters": {"us": ["OTC_SPC"], "eu": ["OTC_FCHI"]},
            "correlation": {"enabled": True, "exclusive_cluster_by_macro": True},
            "macro": {"confluence_conviction_floor": 0.65},
        },
    }
    metrics = base_cluster_metrics()
    metrics.pop("eu_cluster")
    decisions: dict = {}
    propagate_cluster_decisions(
        orch,
        anchor_sym="frxEURUSD",
        direction=TradeDirection.PUT,
        metrics=metrics,
        decisions=decisions,
        cid="C0002",
    )
    assert decisions == {}
    orch.logger.info.assert_called()
