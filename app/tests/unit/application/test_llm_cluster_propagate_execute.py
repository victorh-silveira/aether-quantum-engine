from unittest.mock import MagicMock

from src.application.services.llm.cluster_statarb_select import resolve_statarb_cluster_config_for_tag
from src.application.services.llm.llm_cluster_guards import (
    cluster_entry_allowed,
    cluster_execute_block_reason,
    cluster_execute_flag,
    min_conviction_execute,
)
from src.application.services.llm.llm_cluster_propagate import (
    _rolling_wr_scores,
)
from src.domain.models.trade import TradeDirection
from tests.unit.application.cluster_guard_metrics import base_cluster_metrics


def _base_metrics(**overrides):
    return base_cluster_metrics(**overrides)


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
        is False
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


def test_statarb_cfg_for_macro_tag_overrides_min_abs():
    corr = {"statarb_index_min_abs_z": 0.85}
    macro = {"statarb_min_abs_z_by_tag": {"risk_on": 0.50}}
    cfg = resolve_statarb_cluster_config_for_tag(corr, macro, "risk_on")
    assert cfg["min_abs_z"] == 0.50
    cfg_off = resolve_statarb_cluster_config_for_tag(corr, macro, "risk_off")
    assert cfg_off["min_abs_z"] == 0.85


def test_cluster_execute_blocks_when_us_cluster_weak_on_risk_on():
    orch = MagicMock()
    orch.config = {"llm": {"min_conviction_execute": 0.60}}
    macro = {"confluence_conviction_floor": 0.65, "assert_min_hmm_prob": 0.0}
    corr = {"statarb_require_z_align": True}
    metrics = _base_metrics(
        macro_sentiment="risk_on",
        macro_us_strength_quant=0.40,
        macro_eu_strength_quant=0.90,
        statarb_spreads={"OTC_DJI": -1.0},
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
            target_sym="OTC_DJI",
            llm_cluster_explicit=True,
            index_note="STATARB_BEST leader=OTC_DJI z=-1.00",
        )
        == "macro_or_hmm_veto"
    )
