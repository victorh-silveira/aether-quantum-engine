from unittest.mock import MagicMock

import pytest

from src.domain.risk.risk_stake_calc import calculate_stake_for_manager


def _risk_manager(kelly_config, *, bankroll=10000.0):
    rm = MagicMock()
    rm.config = kelly_config
    rm.kelly_config = {
        **kelly_config["kelly"],
        "fraction": 0.10,
        "super_concordance_booster": 1.5,
        "super_concordance_prob_min": 0.75,
        "super_concordance_hurst_min": 0.55,
        "consensus_penalty_enabled": True,
        "consensus_max_cut": 0.50,
        "consensus_di_weight": 0.30,
        "consensus_cmo_weight": 0.30,
        "consensus_rsi_weight": 0.25,
        "max_stake_pct": 1.0,
        "max_stake_pct_high_conviction": 1.0,
    }
    rm.dlambert_config = {
        **kelly_config.get("dlambert", {}),
    }
    rm.risk_params = {**kelly_config["params"], "stake_min": 1.0, "payout_estimate": 0.95}
    rm.stake_max = 12000.0
    rm.initial_bankroll = bankroll
    rm.total_session_profit = 0.0
    rm.pending_loss = {}
    rm.active_contract_ids = []
    rm.consecutive_losses_linear = 0
    rm.dlambert_unit = 0.0
    rm.logger = MagicMock()
    rm.effective_win_rate = MagicMock(side_effect=lambda _sym, conv, **_kw: float(conv))
    rm._recovery_allowed = MagicMock(return_value=False)
    return rm


def _neutral_metrics():
    return {
        "execute": True,
        "trade_score": 0.62,
        "raw_prob": 0.62,
        "calibrated_prob": 0.61,
        "call_votes": 4,
        "put_votes": 2,
        "indicators": {"hurst": 0.52, "di_diff": 0.05, "cmo": 0.10, "rsi": 0.55},
    }


def _hyper_aligned_metrics():
    return {
        "execute": True,
        "trade_score": 0.82,
        "raw_prob": 0.78,
        "calibrated_prob": 0.79,
        "call_votes": 6,
        "put_votes": 0,
        "indicators": {"hurst": 0.58, "di_diff": 0.12, "cmo": 0.40, "rsi": 0.62},
    }


def _diverged_metrics():
    return {
        "execute": True,
        "trade_score": 0.78,
        "raw_prob": 0.78,
        "calibrated_prob": 0.77,
        "call_votes": 1,
        "put_votes": 5,
        "indicators": {"hurst": 0.60, "di_diff": -0.20, "cmo": -0.50, "rsi": 0.35},
    }


def test_calculate_stake_neutral_without_super_concordance(kelly_config):
    rm = _risk_manager(kelly_config)
    stake = calculate_stake_for_manager(
        rm,
        10000.0,
        "R_10",
        0.62,
        silent=True,
        apply_stop_win=False,
        kwargs={"dl_metrics": _neutral_metrics(), "order_direction": "CALL"},
    )
    hyper_rm = _risk_manager(kelly_config)
    hyper_stake = calculate_stake_for_manager(
        hyper_rm,
        10000.0,
        "R_10",
        0.82,
        silent=True,
        apply_stop_win=False,
        kwargs={"dl_metrics": _hyper_aligned_metrics(), "order_direction": "CALL"},
    )
    assert hyper_stake > stake
    assert hyper_stake == pytest.approx(350.0)


def test_calculate_stake_divergence_blocks_booster_and_applies_consensus(kelly_config):
    rm = _risk_manager(kelly_config)
    aligned_metrics = _hyper_aligned_metrics()
    calculate_stake_for_manager(
        rm,
        10000.0,
        "R_10",
        0.80,
        silent=True,
        apply_stop_win=False,
        kwargs={"dl_metrics": aligned_metrics, "order_direction": "CALL"},
    )
    assert aligned_metrics.get("super_concordance_booster_active") is True

    baseline_rm = _risk_manager(kelly_config)
    baseline_rm.kelly_config["super_concordance_enabled"] = False
    baseline_stake = calculate_stake_for_manager(
        baseline_rm,
        10000.0,
        "R_10",
        0.80,
        silent=True,
        apply_stop_win=False,
        kwargs={"dl_metrics": _hyper_aligned_metrics(), "order_direction": "CALL"},
    )

    diverged_rm = _risk_manager(kelly_config)
    diverged_metrics = _diverged_metrics()
    diverged_stake = calculate_stake_for_manager(
        diverged_rm,
        10000.0,
        "R_10",
        0.80,
        silent=True,
        apply_stop_win=False,
        kwargs={"dl_metrics": diverged_metrics, "order_direction": "CALL"},
    )
    assert diverged_metrics.get("super_concordance_booster_active") is not True
    assert diverged_metrics.get("consensus_entropy_retention", 1.0) < 1.0
    assert diverged_stake <= baseline_stake


def test_calculate_stake_super_concordance_expands_fraction(kelly_config):
    rm = _risk_manager(kelly_config)
    metrics = _hyper_aligned_metrics()
    stake = calculate_stake_for_manager(
        rm,
        10000.0,
        "R_10",
        0.82,
        silent=True,
        apply_stop_win=False,
        kwargs={"dl_metrics": metrics, "order_direction": "CALL"},
    )
    assert metrics.get("super_concordance_booster_active") is True
    assert metrics.get("kelly_fraction_effective") == pytest.approx(0.105)
    assert stake == pytest.approx(350.0)
