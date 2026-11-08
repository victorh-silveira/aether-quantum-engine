"""Pesquisa cronologica sem acesso ao teste na selecao."""

import pytest

from indicator.research_pipeline import SPECS, fit_candidate, run_research


def dataset(n=300):
    close = [(0.01 * ((index % 7) - 3),) * 8 for index in range(n)]
    ohlc = [row + row + tuple(abs(value) for value in row) for row in close]
    labels = [int(index % 7 > 3) for index in range(n)]
    return close, ohlc, labels


def test_five_candidate_budget_and_tree_fit():
    close, ohlc, labels = dataset()
    assert len(SPECS) <= 12
    tree = fit_candidate(ohlc[:100], labels[:100], ("tree", 0.0))
    assert tree.feature_set == "tree"
    assert 0 < tree.tree_left < 1
    assert 0 < tree.tree_right < 1
    assert tree.score(ohlc[0]) != tree.score(ohlc[6])
    logistic = fit_candidate(close[:100], labels[:100], ("close", 0.001))
    assert logistic.feature_set == "close"
    assert len(logistic.weights) == 8


def test_last_twenty_percent_do_not_change_fit_or_calibration():
    close, ohlc, labels = dataset()
    first = run_research(close, ohlc, labels)
    changed = labels[:240] + [1 - label for label in labels[240:]]
    second = run_research(close, ohlc, changed)
    assert first.fitted == second.fitted
    assert first.slope == pytest.approx(second.slope)
    assert first.offset == pytest.approx(second.offset)
    assert first.validation_samples == 60
    assert len(first.candidate_briers) == 5
