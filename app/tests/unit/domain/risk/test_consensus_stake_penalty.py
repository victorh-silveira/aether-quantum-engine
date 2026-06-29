from src.domain.risk.consensus_stake_penalty import consensus_kelly_retention


_CFG = {
    "consensus_penalty_enabled": True,
    "consensus_max_cut": 0.50,
    "consensus_di_weight": 0.35,
    "consensus_cmo_weight": 0.40,
}


def test_c0011_like_divergence_reduces_retention():
    metrics = {
        "call_votes": 1,
        "put_votes": 5,
        "indicators": {"di_diff": 0.01, "cmo": -0.18},
    }
    retention = consensus_kelly_retention(metrics, "CALL", kelly_config=_CFG)
    assert 0.50 <= retention < 1.0


def test_aligned_direction_full_retention():
    metrics = {
        "call_votes": 0,
        "put_votes": 6,
        "indicators": {"di_diff": -0.06, "cmo": -0.71},
    }
    assert consensus_kelly_retention(metrics, "PUT", kelly_config=_CFG) == 1.0


def test_tied_votes_no_penalty():
    metrics = {"call_votes": 3, "put_votes": 3, "indicators": {"cmo": -0.5}}
    assert consensus_kelly_retention(metrics, "CALL", kelly_config=_CFG) == 1.0


def test_disabled_flag():
    metrics = {"call_votes": 1, "put_votes": 5, "indicators": {"cmo": -0.5}}
    disabled = {"consensus_penalty_enabled": False}
    assert consensus_kelly_retention(metrics, "CALL", kelly_config=disabled) == 1.0


def test_call_majority_put_order_penalizes_positive_cmo():
    metrics = {
        "call_votes": 6,
        "put_votes": 1,
        "indicators": {"di_diff": 0.12, "cmo": 0.25},
    }
    retention = consensus_kelly_retention(metrics, "PUT", kelly_config=_CFG)
    assert 0.50 <= retention < 1.0


def test_invalid_order_direction_returns_one():
    metrics = {"call_votes": 1, "put_votes": 5, "indicators": {"cmo": -0.5}}
    assert consensus_kelly_retention(metrics, "HOLD", kelly_config=_CFG) == 1.0
