import pytest

from src.domain.analytics.sample_size_policy import (
    binomial_evidence_z,
    empirical_rate_shrink,
    explore_stake_scale,
    has_underperformance_evidence,
    is_large_sample,
    is_small_sample,
    load_sample_size_policy,
    reset_sample_size_policy_cache,
    sample_reliability,
)
from src.domain.analytics.side_equilibrium import (
    ACTION_HARD_SKIP,
    ACTION_PASS,
    SideCounts,
    SideEquilibriumConfig,
    evaluate_side_equilibrium,
)


@pytest.fixture(autouse=True)
def _reset_policy_cache():
    reset_sample_size_policy_cache()
    yield
    reset_sample_size_policy_cache()


def test_sample_size_policy_ssot_defaults():
    cfg = load_sample_size_policy()
    assert cfg["enabled"] is True
    assert cfg["evidence_n_min"] == 20
    assert cfg["large_n_min"] == 40
    assert cfg["calib_soft_min_n"] == 15
    assert cfg["toxic_side_n_min"] == 8
    assert cfg["explore_stake_scale_floor"] == pytest.approx(0.25)
    assert cfg["z_sig_threshold"] == pytest.approx(1.64)


def test_small_sample_is_noise_until_evidence_n():
    assert is_small_sample(0) is True
    assert is_small_sample(19) is True
    assert is_small_sample(20) is False
    assert is_large_sample(39) is False
    assert is_large_sample(40) is True


def test_reliability_and_shrink_dilute_small_n():
    assert sample_reliability(0) == pytest.approx(0.0)
    assert sample_reliability(20) == pytest.approx(0.5)
    shrunk = empirical_rate_shrink(0.0, n=2, prior=0.55)
    assert shrunk == pytest.approx(0.55 * (1.0 - sample_reliability(2)) + 0.0 * sample_reliability(2))
    assert shrunk > 0.45


def test_underperformance_requires_evidence_n_not_two_losses():
    assert has_underperformance_evidence(0, 2, p0=0.55) is False
    assert has_underperformance_evidence(0, 20, p0=0.55) is True
    assert has_underperformance_evidence(0, 8, p0=0.55, min_n=8) is True
    assert binomial_evidence_z(10, 20, 0.5) == pytest.approx(0.0)


def test_explore_stake_scale_floor_on_cold_start():
    assert explore_stake_scale(0) == pytest.approx(0.25)
    assert explore_stake_scale(20) > explore_stake_scale(0)
    assert explore_stake_scale(40) == pytest.approx(1.0)


def test_side_eq_two_losses_are_insufficient_under_lln_defaults():
    counts = SideCounts(call_n=0, call_wins=0, put_n=2, put_wins=0)
    cfg = SideEquilibriumConfig(n_min_small=8, wr_floor_small=0.40, freq_bias_max_small=0.70)
    decision = evaluate_side_equilibrium(counts, "PUT", config=cfg, regime="small")
    assert decision.action == ACTION_PASS
    assert decision.reason == "small_n_insufficient"


def test_side_eq_hard_skip_needs_significant_wr_when_freq_balanced():
    counts = SideCounts(call_n=20, call_wins=4, put_n=20, put_wins=12)
    cfg = SideEquilibriumConfig(
        n_min_small=8,
        wr_floor_small=0.40,
        freq_bias_max_small=0.90,
        break_even_wr=0.55,
        require_wr_significance=True,
    )
    decision = evaluate_side_equilibrium(counts, "CALL", config=cfg, regime="small")
    assert decision.action == ACTION_HARD_SKIP
    assert "wr_sig" in decision.reason or "small_n" in decision.reason


def test_side_eq_borderline_wr_without_significance_does_not_hard_skip():
    counts = SideCounts(call_n=8, call_wins=3, put_n=8, put_wins=5)
    cfg = SideEquilibriumConfig(
        n_min_small=8,
        wr_floor_small=0.40,
        freq_bias_max_small=0.90,
        break_even_wr=0.55,
        require_wr_significance=True,
    )
    decision = evaluate_side_equilibrium(counts, "CALL", config=cfg, regime="small")
    assert decision.action == ACTION_PASS


def test_sample_size_policy_disabled_branches():
    disabled = {**load_sample_size_policy(), "enabled": False}
    assert is_small_sample(1, policy=disabled) is False
    assert is_large_sample(1, policy=disabled) is True
    assert sample_reliability(0, policy=disabled) == pytest.approx(1.0)
    assert explore_stake_scale(0, policy=disabled) == pytest.approx(1.0)
    assert has_underperformance_evidence(0, 2, p0=0.55, policy=disabled) is True
    assert has_underperformance_evidence(2, 2, p0=0.55, policy=disabled) is False


def test_binomial_evidence_z_degenerate_se():
    from unittest.mock import patch

    assert binomial_evidence_z(0, 0, p0=0.5) == 0.0
    with patch("src.domain.analytics.sample_size_policy.sqrt", return_value=0.0):
        assert binomial_evidence_z(1, 10, p0=0.5) == 0.0


def test_side_wins_unknown_side():
    from src.domain.analytics.side_equilibrium import SideCounts, _side_wins

    assert _side_wins(SideCounts(call_n=2, call_wins=1), "HOLD") == 0


def test_parse_require_wr_significance_override():
    from src.domain.analytics.side_equilibrium import parse_side_equilibrium_config

    cfg = parse_side_equilibrium_config({"require_wr_significance": False})
    assert cfg.require_wr_significance is False


def test_side_eq_wr_without_significance_flag():
    counts = SideCounts(call_n=8, call_wins=1, put_n=2, put_wins=1)
    cfg = SideEquilibriumConfig(
        n_min_small=6,
        wr_floor_small=0.40,
        freq_bias_max_small=0.99,
        require_wr_significance=False,
    )
    decision = evaluate_side_equilibrium(counts, "CALL", config=cfg, regime="small")
    assert decision.action == ACTION_HARD_SKIP
