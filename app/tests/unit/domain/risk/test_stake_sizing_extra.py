import pytest

from src.domain.risk.stake_sizing import _resolve_stop_win_max_stake_pct, resolve_stake_conviction


def test_resolve_stop_win_max_stake_pct_default_one_percent():
    pct = _resolve_stop_win_max_stake_pct({}, {}, 0.95)
    assert pct == pytest.approx(0.01 / 0.95, rel=1e-6)


def test_resolve_stop_win_max_stake_pct_from_stop_win():
    pct = _resolve_stop_win_max_stake_pct({"large_account_stop_win_pct": 4.0}, {}, 0.95)
    assert pct == pytest.approx(0.04 / 0.95, rel=1e-6)


def test_resolve_stop_win_max_stake_pct_explicit_override():
    pct = _resolve_stop_win_max_stake_pct({}, {"stop_win_max_stake_pct": 0.03}, 0.95)
    assert pct == 0.03


def test_resolve_stop_win_max_stake_pct_without_payout():
    pct = _resolve_stop_win_max_stake_pct({"large_account_stop_win_pct": 4.0}, {}, 0.0)
    assert pct == pytest.approx(0.04, rel=1e-6)


def test_resolve_stake_conviction_fallback_raw_side():
    metrics = {"trade_score": 0.40, "raw_prob": 0.52}
    config = {"stake_conviction_min_raw": 0.51, "stop_win_kelly_min_conviction": 0.55}
    res = resolve_stake_conviction(metrics, config)
    assert res == pytest.approx(0.52)
