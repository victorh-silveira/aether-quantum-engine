"""Testes do adapt SCALE por maioria de votos."""

from src.application.services.execution_scale_adapt_majority import (
    adapt_on_majority_votes,
    collect_scale_side_votes,
)
from src.domain.models.trade import TradeDirection


def test_collect_votes_counts_tape_mili_rsi():
    metrics = {
        "scale_tape_consensus": "PUT",
        "scale_mili_dir": "PUT",
        "rsi": 0.40,
    }
    payload = collect_scale_side_votes(metrics, TradeDirection.CALL, include_rsi=True)
    assert payload["scale_vote_call_n"] == 1
    assert payload["scale_vote_put_n"] == 3
    assert payload["scale_vote_winner"] == "PUT"
    assert "rsi:PUT" in payload["scale_vote_sources"]


def test_adapt_majority_returns_none_on_tie():
    metrics = {
        "scale_tape_consensus": "PUT",
        "scale_mili_dir": None,
    }
    cfg = {
        "adapt_on_majority_votes": True,
        "adapt_majority_include_rsi": False,
        "adapt_majority_min_lead": 1,
        "adapt_majority_min_votes": 2,
    }
    assert adapt_on_majority_votes(metrics, TradeDirection.CALL, cfg) is None


def test_adapt_majority_flips_with_lead():
    metrics = {
        "scale_tape_consensus": "PUT",
        "scale_mili_dir": "PUT",
        "rsi": 0.35,
    }
    cfg = {
        "adapt_on_majority_votes": True,
        "adapt_majority_include_rsi": True,
        "adapt_majority_min_lead": 1,
        "adapt_majority_min_votes": 3,
        "adapt_majority_rsi_neutral": 0.5,
    }
    out = adapt_on_majority_votes(metrics, TradeDirection.CALL, cfg)
    assert out == TradeDirection.PUT
    assert metrics["scale_adapt_reason"] == "majority_votes"
