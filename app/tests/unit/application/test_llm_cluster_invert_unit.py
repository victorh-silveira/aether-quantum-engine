from unittest.mock import MagicMock, patch

import pytest

from src.application.services.llm.llm_cluster_invert import (
    apply_cluster_binary_invert,
    flip_binary_direction,
)
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


def test_flip_binary_direction():
    assert flip_binary_direction(TradeDirection.PUT) == TradeDirection.CALL
    assert flip_binary_direction(TradeDirection.CALL) == TradeDirection.PUT
    assert flip_binary_direction(TradeDirection.MULTUP) is None


def test_apply_cluster_binary_invert_skips_non_binary_direction():
    metrics = {"execute": False}
    alt, out, ok = apply_cluster_binary_invert(
        TradeDirection.MULTUP,
        metrics,
        index_note="x",
        anchor_sym="frxEURUSD",
        region_note="",
        conviction=0.7,
    )
    assert ok is False
    assert alt == TradeDirection.MULTUP
    assert out["execute"] is False


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


def test_log_cluster_propagation_results_inverted():
    orch = MagicMock()
    log_cluster_propagation_results(
        orch,
        cid="C0002",
        anchor_sym="frxEURUSD",
        corr_cfg={},
        macro_tag="risk_off",
        active_region="eu",
        us_dir=TradeDirection.PUT,
        eu_dir=TradeDirection.PUT,
        us_note="",
        eu_note="",
        propagated_tags=[],
        blocked_tags=[],
        inverted_tags=["OTC_FTSE[P->C]"],
    )
    assert any("CLUSTER_INVERT" in str(c) for c in orch.logger.info.call_args_list)


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
        corr_cfg={"statarb_require_z_align": False, "cluster_invert_on_block": False},
        active_region="eu",
        exclusive=True,
        macro_tag="risk_off",
        invert_on_block=False,
    )
    assert propagated is None
    assert blocked == "OTC_FCHI[P]:low_conviction"
    assert inverted is None


def test_apply_cluster_binary_invert_sets_execute_and_complement_conviction():
    metrics = {"execute": False, "conviction": 0.70}
    alt, out, ok = apply_cluster_binary_invert(
        TradeDirection.PUT,
        metrics,
        index_note="STATARB_BEST",
        anchor_sym="frxEURUSD",
        region_note="",
        conviction=0.70,
    )
    assert ok is True
    assert alt == TradeDirection.CALL
    assert out["execute"] is True
    assert out["llm_exec_inverted"] is True
    assert out["conviction"] == pytest.approx(0.70)


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
                "cluster_invert_on_block": False,
                "statarb_require_z_align": False,
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


def test_propagate_appends_inverted_tag_from_target_decision():
    orch = MagicMock()
    orch.anchor = "frxEURUSD"
    orch.symbols = ["frxEURUSD", "OTC_FTSE"]
    orch.config = {
        "strategy": {
            "clusters": {"us": ["OTC_SPC"], "eu": ["OTC_FTSE"]},
            "correlation": {"enabled": True, "exclusive_cluster_by_macro": True},
            "macro": {},
        }
    }
    decisions: dict = {}
    allowed = (
        None,
        TradeDirection.PUT,
        [],
        ["OTC_FTSE"],
        "",
        "note",
        set(),
        {"OTC_FTSE"},
    )
    with (
        patch(
            "src.application.services.llm.llm_cluster_propagate._cluster_allowed_sets",
            return_value=allowed,
        ),
        patch(
            "src.application.services.llm.llm_cluster_propagate_region.apply_cluster_target_decision",
            return_value=(None, None, "OTC_FTSE[P->C]", None),
        ) as mock_apply,
    ):
        propagate_cluster_decisions(
            orch,
            anchor_sym="frxEURUSD",
            direction=TradeDirection.PUT,
            metrics=_base_metrics(),
            decisions=decisions,
            cid="C0020",
        )
    mock_apply.assert_called_once()
    assert any("CLUSTER_INVERT" in str(c) for c in orch.logger.info.call_args_list)


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
