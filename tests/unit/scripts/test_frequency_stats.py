"""Valida inferencia por dias, multiplicidade e selecao condicionada aos dados."""

import numpy as np
import pytest

from scripts.analysis.frequency_stats import daily_counts, holm_adjust, infer_ev, select_candidate


def test_days_include_zeros_and_ties():
    """Empates contam como ordens e dias sem trade permanecem no denominador."""
    trades = [{"entry_epoch": 1767272400, "outcome": x} for x in (1, -1, 0)]
    days, counts = daily_counts(trades, "2026-01-01", "2026-01-03")
    assert len(days) == 3
    assert counts.sum(axis=0).tolist() == [1, 1, 1]
    assert np.count_nonzero(counts.sum(axis=1)) == 1


def test_empty_and_nonpositive():
    """Ausencia de trades e ausencia de edge nao aprovam candidatos."""
    assert infer_ev(np.zeros((20, 3), dtype=int))["p_value"] == 1
    result = infer_ev(np.tile([5, 5, 0], (60, 1)), samples=500)
    assert result["ev"] == pytest.approx(-0.15)
    assert result["lower95"] <= result["ev"] + 1e-12
    assert not holm_adjust([result])[0]["statistically_eligible"]


def test_positive_dependent_days_and_holm():
    """Correcao considera toda a familia e dados incompletos impedem selecao."""
    good = infer_ev(np.tile([9, 1, 0], (60, 1)), samples=10000)
    bad = infer_ev(np.tile([1, 9, 0], (60, 1)), samples=10000)
    rows = holm_adjust([good, bad])
    assert rows[0]["statistically_eligible"]
    assert not rows[1]["statistically_eligible"]
    for row in rows:
        row.update(trades_per_day=10, max_drawdown=0.1)
    assert select_candidate(rows, data_qualified=False) is None
    assert select_candidate(rows, data_qualified=True) is good


def test_reproducibility_and_tiebreak():
    """Semente fixa e preferencia de frequencia sao contratos da pesquisa."""
    counts = np.array([[12, 2, 1], [1, 4, 0], [8, 2, 0], [0, 0, 0]] * 20)
    assert infer_ev(counts, samples=500) == infer_ev(counts, samples=500)
    base = {"statistically_eligible": True, "trades_per_day": 2, "lower95": 0.1, "max_drawdown": 0.1}
    faster = dict(base, trades_per_day=3, lower95=0.01, max_drawdown=0.5)
    assert select_candidate([base, faster], data_qualified=True) is faster
