from unittest.mock import MagicMock

import numpy as np

from src.application.services.llm.global_macro_confluence import empty_macro_snapshot
from src.application.services.llm.indicators import IndicatorConfig
from src.application.services.llm.symbol_decision_post import append_entropy_high_note, patch_final_symbol_metrics
from src.application.services.llm.symbol_decision_utils import anchor_llm_decision_complete
from src.domain.models.trade import TradeDirection


def test_append_entropy_high_note_when_threshold_met():
    swing = list(np.linspace(1.0, 2.0, 40))
    runtime = {"indicator_config": IndicatorConfig()}
    out = append_entropy_high_note("base", 0.9, swing, runtime, None)
    assert "ENTROPY_HIGH" in out


def test_anchor_llm_decision_complete_requires_all_tags():
    orch = MagicMock()
    orch.anchor = "frxEURUSD"
    orch.config = {"strategy": {"correlation": {"enabled": True}}}
    ok, tag = anchor_llm_decision_complete(orch, "frxEURUSD", TradeDirection.PUT, TradeDirection.PUT, None)
    assert ok is False
    assert tag == "LLM_EU_CLUSTER_AUSENTE"
    ok2, _ = anchor_llm_decision_complete(
        orch, "frxEURUSD", TradeDirection.PUT, TradeDirection.PUT, TradeDirection.CALL
    )
    assert ok2 is True
    ok3, tag3 = anchor_llm_decision_complete(orch, "frxEURUSD", None, None, None)
    assert ok3 is False
    assert tag3 == "LLM_EURUSD_AUSENTE"
    ok_us, tag_us = anchor_llm_decision_complete(orch, "frxEURUSD", TradeDirection.CALL, None, TradeDirection.PUT)
    assert ok_us is False
    assert tag_us == "LLM_US_CLUSTER_AUSENTE"
    orch.config = {"strategy": {"correlation": {"enabled": False}}}
    ok4, _ = anchor_llm_decision_complete(orch, "frxEURUSD", TradeDirection.CALL, None, None)
    assert ok4 is True


def test_anchor_allows_missing_us_when_quant_flat():
    orch = MagicMock()
    orch.anchor = "frxEURUSD"
    orch.config = {"strategy": {"correlation": {"enabled": True}}}
    snap = empty_macro_snapshot()
    snap = type(snap)(
        us_dir="flat",
        eu_dir="up",
        us_strength=0.0,
        eu_strength=1.0,
        tag="indefinido",
        eurusd_bias=snap.eurusd_bias,
        cluster_status=snap.cluster_status,
        macro_block=snap.macro_block,
        fx_reference_line=snap.fx_reference_line,
        us_parts=snap.us_parts,
        eu_parts=snap.eu_parts,
    )
    ok, tag = anchor_llm_decision_complete(
        orch,
        "frxEURUSD",
        TradeDirection.CALL,
        None,
        TradeDirection.CALL,
        macro_snapshot=snap,
    )
    assert ok is True
    assert tag == ""


def test_patch_final_symbol_metrics():
    snap = empty_macro_snapshot()
    metrics: dict = {}
    patch_final_symbol_metrics(
        metrics,
        execute_flag=True,
        inverted=False,
        llm_http_ms=1.0,
        llm_resp_chars=10,
        llm_direction_from_api=True,
        us_dir=TradeDirection.PUT,
        eu_dir=TradeDirection.CALL,
        macro_snapshot=snap,
        macro_guard=False,
    )
    assert metrics["execute"] is True
    assert metrics["us_cluster"] == "PUT"
    assert metrics["eu_cluster"] == "CALL"
    assert "cluster_index_directions" not in metrics
