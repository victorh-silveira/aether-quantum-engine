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
        statarb_spreads={"1HZ50V": -1.5},
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
            target_sym="1HZ50V",
            llm_cluster_explicit=True,
            index_note="STATARB_BEST leader=1HZ50V z=-1.50",
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
        statarb_spreads={"R_50": -0.55},
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
            target_sym="R_50",
            llm_cluster_explicit=True,
            index_note="STATARB_BEST leader=R_50 z=-0.55 score=0.55",
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
        statarb_spreads={"R_25": -0.35},
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
            target_sym="R_25",
            llm_cluster_explicit=True,
            index_note="STATARB_WEAK leader=R_25 z=-0.35",
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
    orch._last_loss_symbol = "1HZ50V"
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
            target_sym="1HZ50V",
            llm_cluster_explicit=True,
        )
        == "repeat_loss_setup"
    )


def test_cluster_execute_llm_explicit_risk_off_65_uses_global_conviction_floor():
    orch = MagicMock()
    orch.config = {"llm": {"min_conviction_execute": 0.60}}
    macro = {
        "confluence_conviction_floor": 0.60,
        "assert_min_hmm_prob": 0.0,
        "min_conviction_by_tag": {"risk_off": 0.68},
    }
    corr = {"statarb_require_z_align": False}
    metrics = base_cluster_metrics(
        macro_sentiment="risk_off",
        macro_us_strength_quant=0.30,
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
            target_sym="1HZ100V",
            llm_cluster_explicit=True,
        )
        == "allowed"
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
            target_sym="1HZ50V",
            llm_cluster_explicit=True,
        )
        == "allowed"
    )


def test_cluster_execute_statarb_veto_when_not_llm_explicit():
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
        statarb_spreads={"R_25": -0.35},
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
            target_sym="R_25",
            llm_cluster_explicit=False,
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
            target_sym="R_75",
        )
        is False
    )
    assert (
        cluster_execute_flag(
            orch, base_cluster_metrics(), 0.70, None, macro, corr, active_region="eu", target_sym="R_75"
        )
        is False
    )


def test_cluster_propagate_logs_empty_when_cluster_tags_missing():
    orch = MagicMock()
    orch.anchor = "R_100"
    orch.symbols = ["R_100", "R_75"]
    orch.config = {
        "llm": {"min_conviction_execute": 0.60},
        "strategy": {
            "clusters": {"us": ["R_25"], "eu": ["R_75"]},
            "correlation": {"enabled": True, "exclusive_cluster_by_macro": True},
            "macro": {"confluence_conviction_floor": 0.65},
        },
    }
    metrics = base_cluster_metrics()
    metrics.pop("eu_cluster")
    decisions: dict = {}
    propagate_cluster_decisions(
        orch,
        anchor_sym="R_100",
        direction=TradeDirection.PUT,
        metrics=metrics,
        decisions=decisions,
        cid="C0002",
    )
    assert decisions == {}
    orch.logger.info.assert_called()
