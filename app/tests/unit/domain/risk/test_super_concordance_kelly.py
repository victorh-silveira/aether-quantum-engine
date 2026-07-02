import pytest

from src.domain.risk.super_concordance_kelly import (
    apply_super_concordance_kelly_fraction,
    is_unanimous_vote_alignment,
    resolve_order_side_probability,
    resolve_super_concordance_fraction_multiplier,
)


def _hyper_aligned_metrics(**overrides):
    metrics = {
        "calibrated_prob": 0.78,
        "raw_prob": 0.76,
        "call_votes": 6,
        "put_votes": 0,
        "indicators": {"hurst": 0.58},
    }
    metrics.update(overrides)
    if "indicators" in overrides:
        metrics["indicators"] = {**metrics["indicators"], **overrides["indicators"]}
    return metrics


def test_resolve_order_side_probability_call_and_put():
    metrics = {"calibrated_prob": 0.80, "raw_prob": 0.76}
    assert resolve_order_side_probability(metrics, "CALL") == 0.80
    assert resolve_order_side_probability(metrics, "PUT") == pytest.approx(0.24)


def test_is_unanimous_vote_alignment():
    assert is_unanimous_vote_alignment(6, 0, "CALL") is True
    assert is_unanimous_vote_alignment(0, 6, "PUT") is True
    assert is_unanimous_vote_alignment(5, 1, "CALL") is False


def test_resolve_super_concordance_fraction_multiplier_neutral():
    cfg = {"fraction": 0.10, "super_concordance_booster": 1.5}
    metrics = _hyper_aligned_metrics(call_votes=4, put_votes=2)
    mult = resolve_super_concordance_fraction_multiplier(
        metrics,
        "CALL",
        cfg,
        recovery_active=False,
    )
    assert mult == 1.0


def test_resolve_super_concordance_fraction_multiplier_hyper_aligned():
    cfg = {"fraction": 0.10, "super_concordance_booster": 1.5}
    mult = resolve_super_concordance_fraction_multiplier(
        _hyper_aligned_metrics(),
        "CALL",
        cfg,
        recovery_active=False,
    )
    assert mult == 1.5


def test_resolve_super_concordance_fraction_multiplier_skips_recovery():
    cfg = {"super_concordance_booster": 1.5}
    mult = resolve_super_concordance_fraction_multiplier(
        _hyper_aligned_metrics(),
        "CALL",
        cfg,
        recovery_active=True,
    )
    assert mult == 1.0


def test_resolve_order_side_probability_empty_metrics():
    assert resolve_order_side_probability({}, "CALL") == 0.0


def test_is_unanimous_vote_alignment_invalid_order():
    assert is_unanimous_vote_alignment(6, 0, "INVALID") is False
    assert is_unanimous_vote_alignment(0, 6, "CALL") is False


def test_resolve_super_concordance_fraction_multiplier_prob_below_min():
    cfg = {"super_concordance_booster": 1.5, "super_concordance_prob_min": 0.90}
    mult = resolve_super_concordance_fraction_multiplier(
        _hyper_aligned_metrics(),
        "CALL",
        cfg,
        recovery_active=False,
    )
    assert mult == 1.0


def test_apply_super_concordance_kelly_fraction_sets_metrics_flag():
    metrics = _hyper_aligned_metrics()
    cfg = {"fraction": 0.10, "super_concordance_booster": 1.5}
    f_star = apply_super_concordance_kelly_fraction(
        0.20,
        cfg,
        metrics,
        "CALL",
        recovery_active=False,
    )
    assert f_star == pytest.approx(0.20 * 0.10 * 0.40 * 1.5)
    assert metrics["super_concordance_booster_active"] is True
    assert metrics["kelly_fraction_effective"] == pytest.approx(0.06)
