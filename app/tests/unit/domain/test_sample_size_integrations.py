from src.application.services.live_signal_metrics import apply_live_calib_drift_soft
from src.domain.analytics.sample_size_policy import reset_sample_size_policy_cache
from src.domain.risk.kelly_f_star_adjustments import apply_kelly_fraction_scale


def setup_function():
    reset_sample_size_policy_cache()


def teardown_function():
    reset_sample_size_policy_cache()


def test_calib_drift_soft_ignores_tiny_sample():
    metrics = {
        "live_n": 3,
        "live_ece": 0.95,
        "live_wr": 0.0,
        "raw_prob": 0.90,
    }
    assert apply_live_calib_drift_soft(metrics) is False
    assert metrics.get("calib_drift_soft") is False


def test_calib_drift_soft_skips_when_ece_or_consistency_ok():
    metrics = {
        "live_n": 20,
        "live_ece": 0.01,
        "live_wr": 0.80,
        "raw_prob": 0.82,
    }
    assert apply_live_calib_drift_soft(metrics) is False


def test_calib_drift_soft_fires_after_min_n():
    metrics = {
        "live_n": 15,
        "live_ece": 0.95,
        "live_wr": 0.0,
        "raw_prob": 0.90,
    }
    assert apply_live_calib_drift_soft(metrics) is True
    assert metrics.get("calib_drift_soft") is True


def test_explore_kelly_scale_shrinks_cold_start():
    metrics = {"stake_regime": "EXPLORE", "live_n": 0}
    scaled = apply_kelly_fraction_scale(0.05, metrics)
    assert scaled == 0.05 * 0.25
    assert metrics.get("explore_stake_scale") == 0.25


def test_explore_kelly_scale_full_on_large_n():
    metrics = {"stake_regime": "EXPLORE", "live_n": 40}
    scaled = apply_kelly_fraction_scale(0.05, metrics)
    assert scaled == 0.05


def test_primary_toxic_uses_call_side_n():
    from src.application.services.side_equilibrium_helpers import primary_side_is_toxic
    from src.domain.analytics.side_equilibrium import ACTION_HARD_SKIP, SideEquilibriumDecision

    toxic = SideEquilibriumDecision(
        action=ACTION_HARD_SKIP,
        reason="x",
        side_wr=0.10,
        call_n=10,
        call_wins=1,
        put_n=2,
        put_wins=1,
    )
    assert primary_side_is_toxic(toxic) is True


def test_side_eq_recovery_both_hard_non_toxic_soft_keep():
    from unittest.mock import MagicMock, patch

    from src.application.services.side_equilibrium_gate import resolve_direction_with_side_equilibrium
    from src.domain.analytics.side_equilibrium import ACTION_HARD_SKIP, SideEquilibriumDecision
    from src.domain.models.trade import TradeDirection

    orch = MagicMock()
    primary = SideEquilibriumDecision(
        action=ACTION_HARD_SKIP,
        reason="freq",
        side_wr=0.50,
        call_n=8,
        call_wins=4,
        put_n=2,
        put_wins=1,
    )
    alternate = SideEquilibriumDecision(
        action=ACTION_HARD_SKIP,
        reason="freq",
        side_wr=0.50,
        put_n=8,
        put_wins=4,
        call_n=2,
        call_wins=1,
    )
    metrics: dict = {}
    with patch(
        "src.application.services.side_equilibrium_gate.evaluate_proposed_side_equilibrium",
        side_effect=[primary, alternate],
    ):
        chosen = resolve_direction_with_side_equilibrium(
            orch, "R_10", TradeDirection.CALL, metrics, recovery_active=True
        )
    assert chosen == TradeDirection.CALL
    assert metrics.get("side_eq_recovery_both_hard") is True
