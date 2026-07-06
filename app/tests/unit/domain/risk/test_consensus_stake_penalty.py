import pytest

from src.domain.risk.consensus_stake_penalty import (
    _recovery_waives_consensus_penalty,
    apply_neutral_edge_kelly_base,
    apply_turbo_edge_stake,
    consensus_kelly_retention,
    neutral_edge_dynamic_unit,
    turbo_edge_stake_multiplier,
)


_CFG = {
    "consensus_penalty_enabled": True,
    "consensus_max_cut": 0.50,
    "consensus_di_weight": 0.30,
    "consensus_cmo_weight": 0.30,
    "consensus_rsi_weight": 0.25,
    "consensus_entropy_exponent": 2.0,
}


def test_c0011_like_divergence_reduces_retention():
    metrics = {
        "call_votes": 1,
        "put_votes": 5,
        "indicators": {"di_diff": 0.01, "cmo": -0.18, "rsi": 0.36},
    }
    retention = consensus_kelly_retention(metrics, "CALL", kelly_config=_CFG)
    assert 0.50 <= retention < 1.0


def test_convex_penalty_stronger_on_lopsided_votes():
    mild = {
        "call_votes": 2,
        "put_votes": 4,
        "indicators": {"di_diff": 0.02, "cmo": -0.15, "rsi": 0.40},
    }
    severe = {
        "call_votes": 1,
        "put_votes": 5,
        "indicators": {"di_diff": 0.02, "cmo": -0.15, "rsi": 0.40},
    }
    mild_ret = consensus_kelly_retention(mild, "CALL", kelly_config=_CFG)
    severe_ret = consensus_kelly_retention(severe, "CALL", kelly_config=_CFG)
    assert severe_ret < mild_ret


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


def test_consecutive_losses_waives_consensus_penalty_at_high_trade_score():
    metrics = {
        "call_votes": 1,
        "put_votes": 5,
        "trade_score": 0.75,
        "indicators": {"di_diff": 0.01, "cmo": -0.18, "rsi": 0.36},
    }
    retention = consensus_kelly_retention(
        metrics,
        "CALL",
        kelly_config=_CFG,
        consecutive_losses=2,
        pending_loss_total=0.0,
    )
    assert retention == 1.0


def test_recovery_waive_helper_skips_when_not_in_recovery():
    assert (
        _recovery_waives_consensus_penalty(
            {"trade_score": 0.90},
            _CFG,
            consecutive_losses=0,
            pending_loss_total=0.0,
            order_direction="CALL",
        )
        is False
    )


def test_unanimous_votes_waives_consensus_penalty_in_recovery():
    metrics = {
        "call_votes": 0,
        "put_votes": 6,
        "trade_score": 0.55,
        "indicators": {"di_diff": -0.06, "cmo": -0.71, "rsi": 0.36},
    }
    retention = consensus_kelly_retention(
        metrics,
        "PUT",
        kelly_config=_CFG,
        consecutive_losses=0,
        pending_loss_total=653.12,
    )
    assert retention == 1.0
    assert metrics.get("consensus_penalty_recovery_waived") is True


def test_recovery_trade_score_waiver_at_sixty_eight():
    metrics = {
        "call_votes": 1,
        "put_votes": 5,
        "trade_score": 0.68,
        "indicators": {"di_diff": 0.01, "cmo": -0.18, "rsi": 0.36},
    }
    retention = consensus_kelly_retention(
        metrics,
        "CALL",
        kelly_config=_CFG,
        consecutive_losses=1,
        pending_loss_total=0.0,
    )
    assert retention == 1.0
    assert metrics.get("consensus_penalty_recovery_waived") is True


def test_regime_inversion_waives_consensus_penalty_in_recovery():
    metrics = {
        "call_votes": 1,
        "put_votes": 5,
        "direction_inverted": True,
        "universal_regime": "COMPRESSION_TRAP",
        "indicators": {"di_diff": 0.01, "cmo": -0.18, "rsi": 0.36},
    }
    retention = consensus_kelly_retention(
        metrics,
        "PUT",
        kelly_config=_CFG,
        consecutive_losses=1,
        pending_loss_total=120.0,
    )
    assert retention == 1.0
    assert metrics.get("consensus_penalty_regime_inversion_waived") is True


def test_climax_inversion_waives_consensus_penalty_with_linear_losses():
    metrics = {
        "call_votes": 0,
        "put_votes": 6,
        "direction_inverted": True,
        "universal_regime": "CLIMAX_EXHAUSTION",
        "indicators": {"di_diff": -0.06, "cmo": -0.71},
    }
    retention = consensus_kelly_retention(
        metrics,
        "CALL",
        kelly_config=_CFG,
        consecutive_losses=2,
        pending_loss_total=0.0,
    )
    assert retention == 1.0
    assert metrics.get("consensus_penalty_regime_inversion_waived") is True


def test_invalid_order_direction_returns_one():
    metrics = {"call_votes": 1, "put_votes": 5, "indicators": {"cmo": -0.5}}
    assert consensus_kelly_retention(metrics, "HOLD", kelly_config=_CFG) == 1.0


def test_d_squeeze_downgrade_forces_consensus_floor():
    metrics = {"meta_squeeze_downgrade": True}
    retention = consensus_kelly_retention(metrics, "CALL", kelly_config=_CFG)
    assert retention == pytest.approx(0.50)
    assert metrics.get("consensus_penalty_d_squeeze") is True


def test_consensus_stake_floor_forces_retention_floor():
    metrics = {"consensus_stake_floor": True}
    retention = consensus_kelly_retention(metrics, "PUT", kelly_config=_CFG)
    assert retention == pytest.approx(0.50)
    assert metrics.get("consensus_penalty_d_squeeze") is True


def test_neutral_edge_dynamic_unit_at_11k_bankroll():
    assert neutral_edge_dynamic_unit(11000.0) == pytest.approx(16.50)


def test_neutral_regime_preserves_full_retention_without_zscore_penalty():
    metrics = {"edge_expectancy": "NO_EDGE_NEUTRAL", "call_votes": 4, "put_votes": 2, "indicators": {}}
    retention = consensus_kelly_retention(metrics, "CALL", kelly_config=_CFG)
    assert retention == 1.0
    assert metrics.get("consensus_penalty_edge_zscore") is not True


def test_apply_neutral_edge_kelly_base_raises_to_bankroll_pct():
    metrics = {"edge_expectancy": "NO_EDGE_NEUTRAL"}
    base = apply_neutral_edge_kelly_base(2.0, 11000.0, metrics)
    assert base == pytest.approx(16.50)
    assert metrics["session_base_unit"] == pytest.approx(16.50)


def test_apply_neutral_edge_kelly_base_skips_when_kelly_already_higher():
    metrics = {"edge_expectancy": "NO_EDGE_NEUTRAL"}
    base = apply_neutral_edge_kelly_base(25.0, 11000.0, metrics)
    assert base == pytest.approx(25.0)


def test_apply_neutral_edge_kelly_base_skips_on_d_squeeze():
    metrics = {"edge_expectancy": "NO_EDGE_NEUTRAL", "meta_squeeze_downgrade": True}
    base = apply_neutral_edge_kelly_base(1.0, 11000.0, metrics)
    assert base == pytest.approx(1.0)
    assert "session_base_unit" not in metrics


def test_turbo_edge_multiplier_doubles_on_extreme_zscore():
    metrics = {"edge_expectancy": "WIN_EXPECTED", "edge_zscore": 1.6}
    assert turbo_edge_stake_multiplier(metrics) == pytest.approx(2.0)


def test_turbo_edge_multiplier_inactive_below_threshold():
    metrics = {"edge_expectancy": "WIN_EXPECTED", "edge_zscore": 1.2}
    assert turbo_edge_stake_multiplier(metrics) == 1.0


def test_turbo_edge_multiplier_skips_neutral_regime():
    metrics = {"edge_expectancy": "NO_EDGE_NEUTRAL", "edge_zscore": 2.0}
    assert turbo_edge_stake_multiplier(metrics) == 1.0


def test_turbo_edge_multiplier_skips_d_squeeze():
    metrics = {
        "edge_expectancy": "WIN_EXPECTED",
        "edge_zscore": 2.0,
        "consensus_stake_floor": True,
    }
    assert turbo_edge_stake_multiplier(metrics) == 1.0


def test_apply_turbo_edge_stake_doubles_final_stake():
    metrics = {"edge_expectancy": "WIN_EXPECTED", "edge_zscore": 1.8}
    stake = apply_turbo_edge_stake(20.0, metrics)
    assert stake == pytest.approx(40.0)
    assert metrics.get("consensus_turbo_edge_active") is True


def test_apply_turbo_edge_stake_returns_unchanged_without_turbo():
    metrics = {"edge_expectancy": "WIN_EXPECTED", "edge_zscore": 0.9}
    stake = apply_turbo_edge_stake(20.0, metrics)
    assert stake == pytest.approx(20.0)
    assert metrics.get("consensus_turbo_edge_active") is not True


def test_turbo_edge_multiplier_returns_one_for_non_dict():
    assert turbo_edge_stake_multiplier(None) == 1.0
