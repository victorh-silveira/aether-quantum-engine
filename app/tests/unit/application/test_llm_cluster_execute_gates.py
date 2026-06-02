from unittest.mock import MagicMock

from src.application.services.llm.llm_cluster_guards import (
    cluster_entry_allowed,
    cluster_execute_block_reason,
    cluster_execute_flag,
)
from src.application.services.llm.llm_cluster_propagate import propagate_cluster_decisions
from src.domain.models.trade import TradeDirection
from tests.unit.application.cluster_guard_metrics import base_cluster_metrics


def test_cluster_pause_after_loss_does_not_block_llm_explicit():
    orch = MagicMock()
    orch.config = {"llm": {"min_conviction_execute": 0.60}}
    orch._cluster_pause_after_loss_active = True
    macro = {"confluence_conviction_floor": 0.65, "assert_min_hmm_prob": 0.0}
    corr = {"statarb_require_z_align": True}
    metrics = base_cluster_metrics(
        macro_sentiment="divergence_eu_leads",
        macro_eu_strength_quant=0.70,
        statarb_spreads={"OTC_FCHI": -1.5},
        hmm_state=0,
    )
    assert (
        cluster_execute_block_reason(
            orch,
            metrics,
            0.70,
            TradeDirection.PUT,
            macro,
            corr,
            active_region="eu",
            target_sym="OTC_FCHI",
            llm_cluster_explicit=True,
            index_note="STATARB_BEST leader=OTC_FCHI z=-1.50",
        )
        == "allowed"
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


def test_cluster_execute_llm_explicit_skips_statarb_veto():
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
        == "allowed"
    )


def test_cluster_entry_allowed_indefinido_llm_explicit():
    macro = {"confluence_conviction_floor": 0.60, "allowed_execute_tags": ["indefinido"]}
    metrics = base_cluster_metrics(macro_sentiment="indefinido")
    assert cluster_entry_allowed(metrics, macro, active_region="eu", llm_cluster_explicit=True) is True


def test_cluster_entry_allowed_unknown_tag_llm_explicit_returns_true():
    macro = {"confluence_conviction_floor": 0.60}
    metrics = base_cluster_metrics(macro_sentiment="custom_regime")
    assert cluster_entry_allowed(metrics, macro, active_region="us", llm_cluster_explicit=True) is True


def test_cluster_execute_repeat_loss_setup_blocks():
    orch = MagicMock()
    orch.config = {"llm": {"min_conviction_execute": 0.60}}
    orch._cluster_pause_after_loss_active = False
    orch._last_loss_symbol = "OTC_FCHI"
    orch._last_loss_direction = "PUT"
    macro = {"confluence_conviction_floor": 0.60, "assert_min_hmm_prob": 0.0}
    corr = {"statarb_require_z_align": False}
    metrics = base_cluster_metrics(macro_sentiment="divergence_eu_leads")
    assert (
        cluster_execute_block_reason(
            orch,
            metrics,
            0.75,
            TradeDirection.PUT,
            macro,
            corr,
            active_region="eu",
            target_sym="OTC_FCHI",
            llm_cluster_explicit=True,
        )
        == "repeat_loss_setup"
    )


def test_cluster_execute_llm_explicit_divergence_uses_global_conviction_floor():
    orch = MagicMock()
    orch.config = {"llm": {"min_conviction_execute": 0.60}}
    macro = {
        "confluence_conviction_floor": 0.60,
        "assert_min_hmm_prob": 0.0,
        "min_conviction_by_tag": {"divergence_eu_leads": 0.68},
    }
    corr = {"statarb_require_z_align": False}
    metrics = base_cluster_metrics(
        macro_sentiment="divergence_eu_leads",
        macro_us_strength_quant=0.40,
        macro_eu_strength_quant=0.70,
    )
    assert (
        cluster_execute_block_reason(
            orch,
            metrics,
            0.65,
            TradeDirection.CALL,
            macro,
            corr,
            active_region="eu",
            target_sym="OTC_FCHI",
            llm_cluster_explicit=True,
        )
        == "allowed"
    )


def test_cluster_execute_flag_conviction_and_direction_gates():
    orch = MagicMock()
    orch.config = {"llm": {"min_conviction_execute": 0.60}}
    macro = {"confluence_conviction_floor": 0.65, "assert_min_hmm_prob": 0.0}
    corr = {}
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


def test_cluster_execute_m5_trend_misaligned_mr_regime():
    orch = MagicMock()
    orch.config = {"llm": {"min_conviction_execute": 0.60}}
    macro = {"confluence_conviction_floor": 0.60, "assert_min_hmm_prob": 0.0}
    corr = {"statarb_require_m5_trend_align": True}

    # MR Regime: hmm_state = 0. We buy CALL, but M5 trend is "down" -> should block
    metrics = base_cluster_metrics(
        macro_sentiment="risk_on",
        macro_us_strength_quant=0.80,
        statarb_spreads={"OTC_DJI": -1.5},
        hmm_state=0,
    )
    metrics["index_m5_dir_by_symbol"] = {"OTC_DJI": "down"}

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
        )
        == "m5_trend_misaligned"
    )


def test_cluster_execute_m5_trend_misaligned_trending_regime():
    orch = MagicMock()
    orch.config = {"llm": {"min_conviction_execute": 0.60}}
    macro = {"confluence_conviction_floor": 0.60, "assert_min_hmm_prob": 0.0}
    corr = {"statarb_require_m5_trend_align": True}

    # Trending Regime: hmm_state = 1. We buy CALL, but M5 trend is "flat" (not "up") -> should block
    metrics = base_cluster_metrics(
        macro_sentiment="risk_on",
        macro_us_strength_quant=0.80,
        statarb_spreads={"OTC_DJI": 1.5},
        hmm_state=1,
    )
    metrics["index_m5_dir_by_symbol"] = {"OTC_DJI": "flat"}

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
        )
        == "m5_trend_misaligned"
    )
