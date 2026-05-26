from unittest.mock import MagicMock

from src.application.services.llm.llm_cluster_guards import (
    cluster_entry_allowed,
    cluster_execute_flag,
    min_conviction_execute,
)
from src.application.services.llm.llm_cluster_propagate import _rolling_wr_scores, propagate_cluster_decisions
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


def test_rolling_wr_scores_builds_map():
    orch = MagicMock()
    orch.config = {"risk_management": {"kelly": {"dynamic_min_samples": 2}}}
    orch.risk_manager.get_wr_rolling_stats = MagicMock(side_effect=[(0.6, 3), (0.4, 1)])
    scores = _rolling_wr_scores(orch, ["OTC_FCHI", "OTC_GDAXI"], {"statarb_blend_rolling_wr": True})
    assert scores == {"OTC_FCHI": 0.6}


def test_rolling_wr_scores_disabled_returns_none():
    orch = MagicMock()
    assert _rolling_wr_scores(orch, ["OTC_FCHI"], {"statarb_blend_rolling_wr": False}) is None


def test_rolling_wr_scores_without_risk_manager():
    orch = MagicMock()
    orch.risk_manager = None
    assert _rolling_wr_scores(orch, ["OTC_FCHI"], {"statarb_blend_rolling_wr": True}) is None


def test_min_conviction_execute_reads_risk_limits():
    orch = MagicMock()
    orch.config = {
        "llm": {"min_conviction_execute": 0.60},
        "risk_management": {"limits": {"min_conviction_execute": 0.75}},
    }
    assert min_conviction_execute(orch) == 0.75


def test_cluster_entry_allowed_branches():
    macro = {"confluence_conviction_floor": 0.65, "assert_min_hmm_prob": 0.5}
    assert cluster_entry_allowed(_base_metrics(), macro, active_region="eu") is True
    assert cluster_entry_allowed(_base_metrics(hmm_prob=0.2), macro, active_region="eu") is False
    assert (
        cluster_entry_allowed(
            _base_metrics(macro_sentiment="risk_on", macro_us_strength_quant=0.80),
            macro,
            active_region="us",
        )
        is True
    )
    assert (
        cluster_entry_allowed(
            _base_metrics(macro_sentiment="divergence_us_leads", macro_us_strength_quant=0.80),
            macro,
            active_region="us",
        )
        is True
    )
    assert (
        cluster_entry_allowed(
            _base_metrics(macro_sentiment="divergence_eu_leads", macro_eu_strength_quant=0.80),
            macro,
            active_region="eu",
        )
        is True
    )
    assert (
        cluster_entry_allowed(
            _base_metrics(macro_sentiment="indefinido", macro_eu_strength_quant=0.80),
            macro,
            active_region="eu",
        )
        is True
    )
    assert (
        cluster_entry_allowed(
            _base_metrics(macro_sentiment="indefinido", macro_us_strength_quant=0.80, macro_eu_strength_quant=0.10),
            macro,
            active_region="us",
        )
        is True
    )
    assert (
        cluster_entry_allowed(
            _base_metrics(macro_sentiment="indefinido", macro_us_strength_quant=0.80, macro_eu_strength_quant=0.10),
            macro,
            active_region=None,
        )
        is True
    )
    assert cluster_entry_allowed(_base_metrics(macro_sentiment="custom"), macro, active_region="eu") is False
    assert (
        cluster_entry_allowed(
            _base_metrics(macro_eu_strength_quant=0.10),
            macro,
            active_region="eu",
            llm_cluster_explicit=False,
        )
        is False
    )
    assert (
        cluster_entry_allowed(
            _base_metrics(macro_eu_strength_quant=0.10),
            macro,
            active_region="eu",
            llm_cluster_explicit=True,
        )
        is True
    )
    macro_tags = {
        **macro,
        "allowed_execute_tags": ("risk_off",),
    }
    assert cluster_entry_allowed(_base_metrics(macro_sentiment="risk_on"), macro_tags, active_region="us") is False


def test_cluster_execute_flag_skips_z_check_when_disabled():
    orch = MagicMock()
    orch.config = {"llm": {"min_conviction_execute": 0.60}}
    macro = {"confluence_conviction_floor": 0.65, "assert_min_hmm_prob": 0.0}
    metrics = _base_metrics()
    corr = {"statarb_require_z_align": False}
    assert (
        cluster_execute_flag(
            orch,
            metrics,
            0.70,
            TradeDirection.PUT,
            macro,
            corr,
            active_region="eu",
            target_sym="OTC_FCHI",
            llm_cluster_explicit=True,
        )
        is True
    )


def test_cluster_execute_flag_conviction_and_direction_gates():
    orch = MagicMock()
    orch.config = {"llm": {"min_conviction_execute": 0.60}}
    macro = {"confluence_conviction_floor": 0.65, "assert_min_hmm_prob": 0.0}
    corr = {"statarb_require_z_align": True}
    assert (
        cluster_execute_flag(
            orch,
            _base_metrics(),
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
        cluster_execute_flag(orch, _base_metrics(), 0.70, None, macro, corr, active_region="eu", target_sym="OTC_FCHI")
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
    metrics = _base_metrics()
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
