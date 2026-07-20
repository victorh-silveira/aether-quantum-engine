from types import SimpleNamespace

import pytest

from src.application.services.market_audit_log import (
    pop_contract_audit,
    resolve_meta_payoff_zscore,
    resolve_predicted_edge,
    store_contract_audit,
)
from src.domain.models.trade import TradeDirection, TradeStatus


def test_resolve_predicted_edge_prefers_payoff_key():
    assert resolve_predicted_edge({"predicted_payoff_edge": 0.42}) == 0.42


def test_resolve_meta_payoff_zscore_skips_invalid_values():
    assert resolve_meta_payoff_zscore(None) is None
    assert resolve_meta_payoff_zscore({"meta_payoff_edge_zscore": "bad", "edge_zscore": 0.33}) == 0.33
    assert resolve_meta_payoff_zscore({"meta_payoff_edge_zscore": object()}) is None


def test_store_and_pop_contract_audit():
    orch = SimpleNamespace()
    store_contract_audit(
        orch,
        9,
        symbol="R_10",
        direction="CALL",
        edge=0.11,
        meta_payoff_edge_zscore=-0.55,
        raw_prob=0.71,
    )
    sym, direction, edge, z_score, raw_prob = pop_contract_audit(orch, 9)
    assert sym in {"R_10", "R_50"}
    assert direction == "CALL"
    assert edge == 0.11
    assert z_score == pytest.approx(-0.55)
    assert raw_prob == pytest.approx(0.71)


def test_pop_contract_audit_falls_back_to_contract_direction():
    orch = SimpleNamespace()
    contract = SimpleNamespace(direction=TradeDirection.PUT, status=TradeStatus.OPEN)
    sym, direction, edge, z_score, raw_prob = pop_contract_audit(orch, 4, contract=contract, symbol="R_10")
    assert sym in {"R_10", "R_50"}
    assert direction == "PUT"
    assert edge == 0.0
    assert z_score is None
    assert raw_prob is None
