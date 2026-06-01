from unittest.mock import MagicMock

from src.application.services.llm.llm_cluster_logging import log_cluster_propagation_results
from src.application.services.llm.llm_cluster_propagate import propagate_cluster_decisions
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


def test_log_cluster_propagation_results_corrected():
    orch = MagicMock()
    log_cluster_propagation_results(
        orch,
        cid="C0001",
        anchor_sym="frxEURUSD",
        corr_cfg={},
        macro_tag="divergence_us_leads",
        active_region="us",
        us_dir=TradeDirection.PUT,
        eu_dir=TradeDirection.PUT,
        us_note="",
        eu_note="M5_DIR",
        propagated_tags=["OTC_DJI[P]"],
        blocked_tags=[],
        inverted_tags=[],
        corrected_tags=["OTC_DJI[C->P]"],
    )
    assert any("CLUSTER_CORRECT" in str(c) for c in orch.logger.info.call_args_list)


def test_log_cluster_propagation_results_blocked():
    orch = MagicMock()
    log_cluster_propagation_results(
        orch,
        cid="C0001",
        anchor_sym="frxEURUSD",
        corr_cfg={},
        macro_tag="risk_off",
        active_region="eu",
        us_dir=TradeDirection.PUT,
        eu_dir=TradeDirection.PUT,
        us_note="",
        eu_note="STATARB_NO_Z_ALIGN",
        propagated_tags=[],
        blocked_tags=["OTC_FCHI[P]"],
        inverted_tags=[],
    )
    assert any("CLUSTER_BLOCK" in str(c) for c in orch.logger.info.call_args_list)


def test_apply_cluster_target_blocked_without_invert():
    orch = MagicMock()
    orch._cluster_pause_after_loss_active = False
    orch.config = {"llm": {"min_conviction_execute": 0.90}}
    decisions: dict = {}
    propagated, blocked, inverted, corrected = apply_cluster_target_decision(
        orch,
        target_sym="OTC_FCHI",
        target_direction=TradeDirection.PUT,
        index_note="note",
        metrics=_base_metrics(conviction=0.50),
        decisions=decisions,
        anchor_sym="frxEURUSD",
        conviction=0.50,
        macro_cfg={"assert_min_hmm_prob": 0.0, "allowed_execute_tags": ("risk_off",)},
        corr_cfg={
            "statarb_require_z_align": False,
        },
        active_region="eu",
        exclusive=True,
        macro_tag="risk_off",
    )
    assert propagated is None
    assert blocked == "OTC_FCHI[P]:low_conviction"
    assert inverted is None


def test_cluster_propagate_logs_block_without_invert():
    orch = MagicMock()
    orch.anchor = "frxEURUSD"
    orch.symbols = ["frxEURUSD", "OTC_FCHI"]
    orch.config = {
        "llm": {"min_conviction_execute": 0.90},
        "strategy": {
            "clusters": {"us": [], "eu": ["OTC_FCHI"]},
            "correlation": {
                "enabled": True,
                "exclusive_cluster_by_macro": True,
            },
            "macro": {"allowed_execute_tags": ("risk_off",), "assert_min_hmm_prob": 0.0},
        },
    }
    decisions: dict = {}
    propagate_cluster_decisions(
        orch,
        anchor_sym="frxEURUSD",
        direction=TradeDirection.PUT,
        metrics=_base_metrics(conviction=0.50, macro_eu_strength_quant=0.72),
        decisions=decisions,
        cid="C0011",
    )
    assert decisions["OTC_FCHI"]["metrics"]["execute"] is False
    lines = [str(c) for c in orch.logger.info.call_args_list]
    assert any("CLUSTER_BLOCK" in line for line in lines)


def test_cluster_propagate_logs_block_when_risk_off_weak_eu():
    orch = MagicMock()
    orch._cluster_pause_after_loss_active = False
    orch.anchor = "frxEURUSD"
    orch.symbols = ["frxEURUSD", "OTC_FCHI"]
    orch.config = {
        "llm": {"min_conviction_execute": 0.60},
        "strategy": {
            "clusters": {"us": ["OTC_SPC"], "eu": ["OTC_FCHI"]},
            "correlation": {"enabled": True, "exclusive_cluster_by_macro": True},
            "macro": {"assert_min_hmm_prob": 0.0, "allowed_execute_tags": ("risk_off",)},
        },
    }
    decisions: dict = {}
    propagate_cluster_decisions(
        orch,
        anchor_sym="frxEURUSD",
        direction=TradeDirection.PUT,
        metrics=_base_metrics(macro_eu_strength_quant=0.40),
        decisions=decisions,
        cid="C0003",
    )
    assert decisions["OTC_FCHI"]["direction"] == TradeDirection.PUT
    assert decisions["OTC_FCHI"]["metrics"]["execute"] is False
    assert decisions["OTC_FCHI"]["metrics"]["llm_block_reason"] == "macro_or_hmm_veto"
    lines = [str(c) for c in orch.logger.info.call_args_list]
    assert any("CLUSTER_BLOCK" in line for line in lines)


def test_cluster_execute_ignores_anchor_macro_execute_false():
    orch = MagicMock()
    orch.anchor = "frxEURUSD"
    orch.symbols = ["frxEURUSD", "OTC_FCHI", "OTC_GDAXI"]
    orch.config = {
        "llm": {"min_conviction_execute": 0.60},
        "strategy": {
            "clusters": {"us": ["OTC_SPC"], "eu": ["OTC_FCHI", "OTC_GDAXI"]},
            "correlation": {
                "enabled": True,
                "exclusive_cluster_by_macro": True,
                "best_symbol_only": True,
                "statarb_index_select_enabled": True,
                "statarb_blend_rolling_wr": False,
            },
            "macro": {
                "confluence_conviction_floor": 0.65,
                "assert_min_hmm_prob": 0.58,
                "allowed_execute_tags": ("risk_off",),
            },
        },
    }
    decisions: dict = {}
    metrics = _base_metrics(execute=False, macro_eu_strength_quant=0.72)
    propagate_cluster_decisions(
        orch,
        anchor_sym="frxEURUSD",
        direction=TradeDirection.PUT,
        metrics=metrics,
        decisions=decisions,
        cid="C0001",
    )
    cluster_syms = [s for s in decisions if s != "frxEURUSD"]
    assert cluster_syms
    assert any(decisions[s]["metrics"]["execute"] for s in cluster_syms)
