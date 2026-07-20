from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.application.services.execution_quality_gate_cluster import quality_conviction_suspends_cluster
from src.application.services.execution_symbols_overdrive import (
    _decisions_ranked_pool,
    _meta_payoff_edge_zscore,
    _overdrive_conviction_eligible,
    try_volatility_overdrive_override,
    volatility_overdrive_unblocks_cluster,
)
from src.domain.models.trade import TradeDirection
from tests.market_symbols import ALT_SYMBOL, ANCHOR


def _orch_stub():
    return SimpleNamespace(
        config={"orchestrator": {"execution": {"quality_gate": {}}}},
        risk_manager=SimpleNamespace(
            consecutive_losses_linear=0,
            pending_loss={},
            pending_loss_total=lambda: 0.0,
        ),
    )


def test_meta_payoff_edge_zscore_reads_primary_field():
    assert _meta_payoff_edge_zscore({"meta_payoff_edge_zscore": 0.33}) == pytest.approx(0.33)


def test_meta_payoff_edge_zscore_reads_edge_zscore_when_meta_not_applied():
    assert _meta_payoff_edge_zscore({"edge_zscore": 0.21}) == pytest.approx(0.21)


def test_meta_payoff_edge_zscore_returns_none_when_fallback_missing():
    assert _meta_payoff_edge_zscore({}) is None


def test_overdrive_returns_none_for_single_candidate():
    orch = _orch_stub()
    ranked = [
        (ALT_SYMBOL, TradeDirection.PUT, {"raw_prob": 0.50, "trade_score": 0.90}),
    ]
    assert try_volatility_overdrive_override(orch, ranked) is None


def test_overdrive_returns_none_when_leader_not_neutral():
    orch = _orch_stub()
    ranked = [
        (ALT_SYMBOL, TradeDirection.PUT, {"raw_prob": 0.40, "trade_score": 0.90, "meta_payoff_edge_zscore": 0.1}),
        (ANCHOR, TradeDirection.PUT, {"raw_prob": 0.31, "trade_score": 0.70, "exec_direction": "PUT"}),
    ]
    assert try_volatility_overdrive_override(orch, ranked) is None


def test_overdrive_returns_none_when_alternate_lacks_conviction():
    orch = _orch_stub()
    ranked = [
        (ALT_SYMBOL, TradeDirection.PUT, {"raw_prob": 0.50, "trade_score": 0.90, "meta_payoff_edge_zscore": 0.1}),
        (ANCHOR, TradeDirection.PUT, {"raw_prob": 0.48, "trade_score": 0.70, "meta_payoff_edge_zscore": 0.1}),
    ]
    assert try_volatility_overdrive_override(orch, ranked) is None


def test_overdrive_uses_edge_zscore_fallback():
    orch = _orch_stub()
    ranked = [
        (ALT_SYMBOL, TradeDirection.PUT, {"raw_prob": 0.50, "trade_score": 0.90}),
        (
            ANCHOR,
            TradeDirection.PUT,
            {
                "raw_prob": 0.31,
                "trade_score": 0.70,
                "exec_direction": "PUT",
                "edge_zscore": 0.05,
                "predicted_payoff_edge": 0.05,
                "meta_classifier_applied": True,
            },
        ),
    ]
    redirect = try_volatility_overdrive_override(orch, ranked)
    assert redirect is None


def test_volatility_overdrive_unblocks_cluster_returns_false_for_empty_pool():
    orch = _orch_stub()
    assert volatility_overdrive_unblocks_cluster(orch, {}) is False


def test_overdrive_unblocks_cluster_skips_malformed_decision_entries(orch_ready):
    orch = orch_ready
    decisions = {
        ALT_SYMBOL: "invalid",
        ANCHOR: {
            "metrics": {
                "raw_prob": 0.31,
                "trade_score": 0.70,
                "exec_direction": "PUT",
                "meta_payoff_edge_zscore": 0.55,
                "predicted_payoff_edge": 0.05,
                "meta_classifier_applied": True,
                "deploy_ok": True,
            }
        },
    }
    assert quality_conviction_suspends_cluster(orch, decisions) is False


def test_overdrive_redirects_when_leader_fails_quality_and_alt_has_conviction():
    orch = _orch_stub()
    ranked = [
        (ALT_SYMBOL, TradeDirection.PUT, {"raw_prob": 0.50, "trade_score": 0.90}),
        (ANCHOR, TradeDirection.PUT, {"raw_prob": 0.31, "trade_score": 0.70, "edge_zscore": 0.20}),
    ]
    with patch(
        "src.application.services.execution_symbols_overdrive._passes_dynamic_quality_gate",
        return_value=False,
    ):
        redirect = try_volatility_overdrive_override(orch, ranked)
    assert redirect is not None
    assert redirect[0] == ANCHOR
    assert redirect[2]["volatility_overdrive_selected"] is True
    assert redirect[2]["volatility_overdrive_ignored_symbol"] == ALT_SYMBOL
    assert "volatility_overdrive_conviction" in redirect[2]
    assert ranked[0][2]["volatility_overdrive_bypassed"] is True


def test_overdrive_unblocks_cluster_with_decision_pool_and_patch():
    orch = _orch_stub()
    decisions = {
        ALT_SYMBOL: {"metrics": {"raw_prob": 0.50, "trade_score": 0.90, "market_decision_score_override": 1.0}},
        ANCHOR: {"metrics": {"raw_prob": 0.31, "trade_score": 0.70, "exec_direction": "PUT", "edge_zscore": 0.20}},
    }
    with patch(
        "src.application.services.execution_symbols_overdrive._passes_dynamic_quality_gate",
        return_value=False,
    ):
        assert volatility_overdrive_unblocks_cluster(orch, decisions) is True


def test_overdrive_conviction_eligible_rejects_low_margin():
    assert _overdrive_conviction_eligible({"raw_prob": 0.51, "meta_payoff_edge_zscore": 0.4}) is False


def test_overdrive_returns_none_when_all_alternates_ineligible():
    orch = _orch_stub()
    ranked = [
        (ALT_SYMBOL, TradeDirection.PUT, {"raw_prob": 0.50, "trade_score": 0.90}),
        (ANCHOR, TradeDirection.PUT, {"raw_prob": 0.49, "trade_score": 0.70, "edge_zscore": 0.20}),
    ]
    with patch(
        "src.application.services.execution_symbols_overdrive._passes_dynamic_quality_gate",
        return_value=False,
    ):
        assert try_volatility_overdrive_override(orch, ranked) is None


def test_decisions_ranked_pool_skips_non_dict_entry_and_non_dict_metrics():
    decisions = {
        "BAD_A": "invalid",
        "BAD_B": {"metrics": "invalid"},
        "GOOD": {"metrics": {"raw_prob": 0.60, "trade_score": 0.60, "exec_direction": "CALL"}},
    }
    pool = _decisions_ranked_pool(decisions)
    assert len(pool) == 1
    assert pool[0][0] == "GOOD"
