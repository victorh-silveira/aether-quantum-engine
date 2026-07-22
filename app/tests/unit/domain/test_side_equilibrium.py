import pytest

from src.domain.analytics.side_equilibrium import (
    ACTION_HARD_SKIP,
    ACTION_PASS,
    ACTION_SOFT,
    SideCounts,
    SideEquilibriumConfig,
    binomial_z_vs_p,
    evaluate_side_equilibrium,
    parse_side_equilibrium_config,
)


def test_parse_side_equilibrium_config_defaults():
    cfg = parse_side_equilibrium_config(None)
    assert cfg.enabled is True
    assert cfg.small_window == 12
    assert cfg.n_min_small == 3
    assert cfg.n_min_large == 40
    assert cfg.freq_bias_max_small == pytest.approx(0.70)
    assert parse_side_equilibrium_config({"n_min_small": 1}).n_min_small == 2


def test_small_n_hard_skip_on_hot_losing_call_side():
    counts = SideCounts(call_n=8, call_wins=2, put_n=2, put_wins=1)
    cfg = SideEquilibriumConfig(n_min_small=6, wr_floor_small=0.40, freq_bias_max_small=0.70)
    decision = evaluate_side_equilibrium(counts, "CALL", config=cfg, regime="small")
    assert decision.action == ACTION_HARD_SKIP
    assert "small_n" in decision.reason


def test_small_n_hard_skip_after_two_put_losses():
    counts = SideCounts(call_n=0, call_wins=0, put_n=2, put_wins=0)
    cfg = SideEquilibriumConfig(n_min_small=2, wr_floor_small=0.40, freq_bias_max_small=0.70)
    decision = evaluate_side_equilibrium(counts, "PUT", config=cfg, regime="small")
    assert decision.action == ACTION_HARD_SKIP
    assert decision.freq_bias == pytest.approx(1.0)
    assert decision.side_wr == pytest.approx(0.0)


def test_small_n_pass_when_insufficient_samples():
    counts = SideCounts(call_n=1, call_wins=0, put_n=0, put_wins=0)
    cfg = SideEquilibriumConfig(n_min_small=2)
    decision = evaluate_side_equilibrium(counts, "CALL", config=cfg, regime="small")
    assert decision.action == ACTION_PASS
    assert decision.reason == "small_n_insufficient"


def test_large_n_soft_penalty_without_direction_flip():
    counts = SideCounts(call_n=60, call_wins=24, put_n=40, put_wins=22)
    cfg = SideEquilibriumConfig(n_min_large=40, wr_floor_large=0.48, kelly_mult_soft=0.55, margin_boost_soft=0.03)
    decision = evaluate_side_equilibrium(counts, "CALL", config=cfg, regime="large")
    assert decision.action == ACTION_SOFT
    assert decision.kelly_mult == pytest.approx(0.55)
    assert decision.margin_boost == pytest.approx(0.03)


def test_put_side_unaffected_when_call_is_imbalanced():
    counts = SideCounts(call_n=10, call_wins=2, put_n=2, put_wins=2)
    cfg = SideEquilibriumConfig(n_min_small=6, freq_bias_max_small=0.70, wr_floor_small=0.40)
    call_decision = evaluate_side_equilibrium(counts, "CALL", config=cfg, regime="small")
    put_decision = evaluate_side_equilibrium(counts, "PUT", config=cfg, regime="small")
    assert call_decision.action == ACTION_HARD_SKIP
    assert put_decision.action == ACTION_PASS


def test_side_counts_unknown_side_and_empty_wr():
    counts = SideCounts()
    assert counts.wr("HOLD") is None
    assert counts.side_n("HOLD") == 0
    assert counts.freq_share("CALL") == pytest.approx(0.5)
    assert counts.wr("PUT") is None


def test_binomial_z_vs_p_edges():
    assert binomial_z_vs_p(0, 0) == 0.0
    assert binomial_z_vs_p(5, 10) == pytest.approx(0.0)
    assert abs(binomial_z_vs_p(10, 10, p0=0.5)) > 1.0


def test_evaluate_disabled_and_invalid_side():
    counts = SideCounts(call_n=8, call_wins=4, put_n=4, put_wins=2)
    disabled = SideEquilibriumConfig(enabled=False)
    assert evaluate_side_equilibrium(counts, "CALL", config=disabled, regime="small").reason == "disabled"
    assert (
        evaluate_side_equilibrium(counts, "HOLD", config=SideEquilibriumConfig(), regime="small").action == ACTION_PASS
    )


def test_small_n_freq_only_and_pass_when_healthy():
    hot = SideCounts(call_n=8, call_wins=5, put_n=2, put_wins=1)
    cfg = SideEquilibriumConfig(n_min_small=6, wr_floor_small=0.20, freq_bias_max_small=0.70, break_even_wr=0.90)
    decision = evaluate_side_equilibrium(hot, "CALL", config=cfg, regime="small")
    assert decision.action == ACTION_HARD_SKIP
    assert decision.reason == "side_imbalance_small_n_freq"
    healthy = SideCounts(call_n=6, call_wins=5, put_n=6, put_wins=4)
    ok = evaluate_side_equilibrium(
        healthy,
        "CALL",
        config=SideEquilibriumConfig(n_min_small=6, wr_floor_small=0.40, freq_bias_max_small=0.90),
        regime="small",
    )
    assert ok.action == ACTION_PASS
    assert ok.reason == "ok"


def test_large_n_insufficient_and_pass_when_balanced():
    small = SideCounts(call_n=5, call_wins=2, put_n=5, put_wins=2)
    assert (
        evaluate_side_equilibrium(small, "CALL", config=SideEquilibriumConfig(n_min_large=40), regime="large").reason
        == "large_n_insufficient"
    )
    balanced = SideCounts(call_n=50, call_wins=30, put_n=50, put_wins=28)
    ok = evaluate_side_equilibrium(
        balanced,
        "CALL",
        config=SideEquilibriumConfig(n_min_large=40, wr_floor_large=0.48, freq_bias_max_large=0.70),
        regime="large",
    )
    assert ok.action == ACTION_PASS
