import pytest

from src.application.services.market_audit_log import (
    pop_contract_audit,
    resolve_meta_payoff_zscore,
    resolve_predicted_edge,
    store_contract_audit,
)


class TestResolvePredictedEdge:
    def test_returns_edge_from_calibrated_prob(self):
        edge = resolve_predicted_edge({"calibrated_prob": 0.7})
        assert edge == pytest.approx((0.7 * 1.95) - 1.0)

    def test_falls_back_to_raw_prob(self):
        edge = resolve_predicted_edge({"raw_prob": 0.65})
        assert edge == pytest.approx((0.65 * 1.95) - 1.0)

    def test_defaults_to_0_5_when_no_prob_key(self):
        edge = resolve_predicted_edge({"some_key": 1.0})
        assert edge == pytest.approx((0.5 * 1.95) - 1.0)

    def test_returns_0_for_non_dict_input(self):
        assert resolve_predicted_edge(None) == 0.0
        assert resolve_predicted_edge("invalid") == 0.0

    def test_returns_0_when_prob_is_none(self):
        edge = resolve_predicted_edge({"calibrated_prob": None, "raw_prob": None})
        assert edge == 0.0

    def test_uses_dominant_prob_when_below_0_5(self):
        edge = resolve_predicted_edge({"raw_prob": 0.3})
        assert edge == pytest.approx((0.7 * 1.95) - 1.0)

    def test_accepts_custom_payout(self):
        edge = resolve_predicted_edge({"calibrated_prob": 0.8}, payout=0.90)
        assert edge == pytest.approx((0.8 * 1.90) - 1.0)


class TestResolveMetaPayoffZscore:
    def test_returns_0_for_none(self):
        assert resolve_meta_payoff_zscore(None) == 0.0

    def test_returns_0_for_non_dict(self):
        assert resolve_meta_payoff_zscore("bad") == 0.0

    def test_reads_meta_payoff_edge_zscore(self):
        assert resolve_meta_payoff_zscore({"meta_payoff_edge_zscore": 1.5}) == 1.5

    def test_falls_back_to_edge_zscore(self):
        assert resolve_meta_payoff_zscore({"edge_zscore": -0.75}) == -0.75

    def test_returns_0_when_missing(self):
        assert resolve_meta_payoff_zscore({}) == 0.0

    def test_reads_float_string_value(self):
        assert resolve_meta_payoff_zscore({"meta_payoff_edge_zscore": "0.42"}) == 0.42


class TestStoreAndPopContractAudit:
    def test_store_and_pop_roundtrip(self):
        cid = "999"
        data = {"symbol": "R_10", "direction": "CALL", "edge": 0.11, "meta_payoff_edge_zscore": -0.55, "raw_prob": 0.71}
        store_contract_audit(cid, data)
        assert pop_contract_audit(cid) == data

    def test_pop_returns_empty_dict_for_missing(self):
        assert pop_contract_audit("nonexistent") == {}

    def test_pop_only_once(self):
        cid = "once"
        store_contract_audit(cid, {"edge": 0.5})
        assert pop_contract_audit(cid) == {"edge": 0.5}
        assert pop_contract_audit(cid) == {}

    def test_store_ignores_empty_id(self):
        store_contract_audit("", {"edge": 0.5})
