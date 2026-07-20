import pytest

from src.application.services.deep_learning.dl_indicator_prob_blend import (
    blend_prob_with_indicator_consensus,
    indicator_vote_share,
)


def test_indicator_vote_share_normalizes():
    call_share, put_share, total = indicator_vote_share(6, 2)
    assert total == 8
    assert call_share == pytest.approx(0.75)
    assert put_share == pytest.approx(0.25)


def test_blend_boosts_call_on_strong_consensus():
    blended, delta, mode = blend_prob_with_indicator_consensus(0.496, call_votes=7, put_votes=2, adx=0.22)
    assert mode == "call_consensus"
    assert delta > 0.0
    assert blended >= 0.52


def test_blend_boosts_put_on_strong_consensus():
    blended, delta, mode = blend_prob_with_indicator_consensus(0.504, call_votes=2, put_votes=7, adx=0.22)
    assert mode == "put_consensus"
    assert delta < 0.0
    assert blended <= 0.48


def test_blend_skips_when_adx_weak():
    blended, delta, mode = blend_prob_with_indicator_consensus(0.496, call_votes=8, put_votes=1, adx=0.08)
    assert mode == "adx_weak"
    assert delta == 0.0
    assert blended == pytest.approx(0.496)


def test_blend_skips_without_majority():
    blended, delta, mode = blend_prob_with_indicator_consensus(0.50, call_votes=4, put_votes=4, adx=0.25)
    assert mode == "no_majority"
    assert delta == 0.0
    assert blended == pytest.approx(0.50)


def test_blend_skips_when_tcn_already_decisive():
    blended, delta, mode = blend_prob_with_indicator_consensus(0.80, call_votes=8, put_votes=1, adx=0.30)
    assert mode == "tcn_decisive"
    assert delta == 0.0
    assert blended == pytest.approx(0.80)
