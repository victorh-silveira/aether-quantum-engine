"""Testes do data loader de backtest (sem cache)."""

from scripts.backtest.data_loader import _align_series_lengths, resolve_bar_counts


def test_resolve_bar_counts_days():
    config = {"strategy": {"macro": {"statarb_lookback": 30, "cluster_bars": 8}}}
    m15, m5 = resolve_bar_counts(config, days=5, bars=None)
    assert m15 == 5 * 96 + 30 + 5
    assert m5 > m15


def test_align_series_lengths():
    m15 = {"A": list(range(100)), "B": list(range(80))}
    m5 = {"A": list(range(300)), "B": list(range(200))}
    m15_out, m5_out, n = _align_series_lengths(m15, m5)
    assert n == 80
    assert len(m15_out["A"]) == 80
    assert m15_out["A"][0] == 20
