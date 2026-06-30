import pytest

from src.domain.risk.stake_sizing import (
    consensus_entropy_applies_min_stake,
    consensus_entropy_kelly_retention,
    consensus_vote_agreement,
)


def test_consensus_vote_agreement():
    assert consensus_vote_agreement(1, 5, "CALL") == pytest.approx(1 / 6)
    assert consensus_vote_agreement(6, 1, "PUT") == pytest.approx(1 / 7)
    assert consensus_vote_agreement(0, 0, "CALL") == 1.0


def test_consensus_entropy_applies_min_stake_at_floor():
    cfg = {"consensus_penalty_enabled": True, "consensus_max_cut": 0.50}
    assert consensus_entropy_applies_min_stake(0.50, cfg) is True
    assert consensus_entropy_applies_min_stake(0.75, cfg) is False
    assert consensus_entropy_applies_min_stake(0.50, {"consensus_penalty_enabled": False}) is False


def test_consensus_entropy_kelly_retention_put_rsi_overbought():
    cfg = {
        "consensus_penalty_enabled": True,
        "consensus_max_cut": 0.50,
        "consensus_di_weight": 0.30,
        "consensus_cmo_weight": 0.30,
        "consensus_rsi_weight": 0.25,
        "consensus_entropy_exponent": 2.0,
    }
    metrics = {
        "call_votes": 6,
        "put_votes": 1,
        "indicators": {"di_diff": -0.10, "cmo": 0.40, "rsi": 0.72},
    }
    retention = consensus_entropy_kelly_retention(metrics, "PUT", kelly_config=cfg)
    assert 0.50 <= retention < 1.0


def test_consensus_entropy_kelly_retention_invalid_order():
    assert consensus_entropy_kelly_retention({}, "HOLD") == 1.0
