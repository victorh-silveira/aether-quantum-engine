import pytest

from src.domain.risk.dlambert_sizing import (
    GEOMETRIC_MARTINGALE_BASE,
    REDIS_DLAMBERT_LINEAR_LOSSES_KEY,
    REDIS_DLAMBERT_UNIT_KEY,
    _resolve_override_value,
    dlambert_enabled,
    dlambert_log_suffix,
    effective_martingale_base,
    geometric_martingale_stake,
    martingale_recovery_active,
    resolve_dlambert_stake,
    resolve_dlambert_unit,
)


def test_redis_keys():
    assert REDIS_DLAMBERT_UNIT_KEY == "session:current:dlambert_unit"
    assert REDIS_DLAMBERT_LINEAR_LOSSES_KEY == "session:current:consecutive_losses_linear"


def test_geometric_martingale_base_constant():
    assert pytest.approx(2.0) == GEOMETRIC_MARTINGALE_BASE


def test_geometric_martingale_stake_doubles_each_loss():
    assert geometric_martingale_stake(50.0, 0) == pytest.approx(50.0)
    assert geometric_martingale_stake(50.0, 1) == pytest.approx(100.0)
    assert geometric_martingale_stake(50.0, 2) == pytest.approx(200.0)
    assert geometric_martingale_stake(50.0, 3) == pytest.approx(400.0)


def test_geometric_martingale_stake_losses_three_is_base_times_eight():
    assert geometric_martingale_stake(10.0, 3) == pytest.approx(80.0)


def test_geometric_martingale_stake_clamps_negative_inputs():
    assert geometric_martingale_stake(-5.0, 4) == pytest.approx(0.0)
    assert geometric_martingale_stake(20.0, -3) == pytest.approx(20.0)


def test_geometric_martingale_stake_has_no_ceiling():
    deep = geometric_martingale_stake(50.0, 12)
    assert deep == pytest.approx(50.0 * (2.0**12))
    assert deep > 100000.0


def test_dlambert_enabled_defaults_true_and_respects_flag():
    assert dlambert_enabled({}) is True
    assert dlambert_enabled({"dlambert_enabled": False}) is False


def test_resolve_dlambert_unit_captures_first_kelly():
    class RM:
        dlambert_unit = 0.0
        dlambert_config = {"dlambert_unit_override": None}

    rm = RM()
    assert resolve_dlambert_unit(42.5, rm) == pytest.approx(42.5)
    assert rm.dlambert_unit == pytest.approx(42.5)
    assert resolve_dlambert_unit(99.0, rm) == pytest.approx(42.5)


def test_resolve_dlambert_unit_override():
    class RM:
        dlambert_unit = 0.0
        dlambert_config = {"dlambert_unit_override": 30.0}

    rm = RM()
    assert resolve_dlambert_unit(50.0, rm) == pytest.approx(30.0)


def test_resolve_dlambert_unit_invalid_override():
    class RM:
        dlambert_unit = 0.0
        dlambert_config = {"dlambert_unit_override": "bad"}

    rm = RM()
    assert resolve_dlambert_unit(25.0, rm) == pytest.approx(25.0)


def test_resolve_dlambert_unit_zero_kelly_returns_zero():
    class RM:
        dlambert_unit = 0.0
        dlambert_config = {}

    assert resolve_dlambert_unit(0.0, RM()) == pytest.approx(0.0)


def test_resolve_dlambert_stake_kelly_when_flat():
    class RM:
        dlambert_unit = 0.0
        dlambert_config = {}

    stake, tag = resolve_dlambert_stake(
        recovery_active=False,
        bankroll=10000.0,
        kelly_base=55.0,
        dlambert_config={"dlambert_enabled": True},
        rm=RM(),
        consecutive_losses_linear=0,
    )
    assert tag == "KELLY"
    assert stake == pytest.approx(55.0)


def test_resolve_override_value_handles_invalid_and_non_positive_values():
    assert _resolve_override_value({}) == pytest.approx(0.0)
    assert _resolve_override_value({"dlambert_unit_override": "bad"}) == pytest.approx(0.0)
    assert _resolve_override_value({"dlambert_unit_override": -2.0}) == pytest.approx(0.0)


def test_martingale_recovery_active_pending_or_linear_losses():
    assert martingale_recovery_active(recovery_active=False, pending_total=93.19, consecutive_losses_linear=0)
    assert martingale_recovery_active(recovery_active=False, pending_total=0.0, consecutive_losses_linear=3)
    assert not martingale_recovery_active(recovery_active=False, pending_total=0.0, consecutive_losses_linear=0)


def test_effective_martingale_base_anchors_above_compressed_kelly():
    class RM:
        dlambert_unit = 10.0
        dlambert_config = {"dlambert_unit_override": 8.0}

    assert effective_martingale_base(1.0, RM(), {"dlambert_unit_override": 8.0}) == pytest.approx(10.0)


def test_resolve_dlambert_stake_c0017_anchors_unit_times_eight_despite_kelly_collapse():
    class RM:
        dlambert_unit = 10.0
        dlambert_config = {}

    stake, tag = resolve_dlambert_stake(
        recovery_active=False,
        bankroll=10000.0,
        kelly_base=1.0,
        dlambert_config={"dlambert_enabled": True},
        rm=RM(),
        consecutive_losses_linear=3,
        pending_total=93.19,
    )
    assert tag == "D'ALEMBERT"
    assert stake == pytest.approx(80.0)


def test_resolve_dlambert_stake_geometric_in_recovery():
    class RM:
        dlambert_unit = 0.0
        dlambert_config = {}

    stake, tag = resolve_dlambert_stake(
        recovery_active=True,
        bankroll=10000.0,
        kelly_base=50.0,
        dlambert_config={"dlambert_enabled": True},
        rm=RM(),
        consecutive_losses_linear=3,
        pending_total=400.0,
    )
    assert tag == "D'ALEMBERT"
    assert stake == pytest.approx(400.0)


def test_resolve_dlambert_stake_progresses_without_bankroll_cap():
    class RM:
        dlambert_unit = 0.0
        dlambert_config = {}

    stake, tag = resolve_dlambert_stake(
        recovery_active=True,
        bankroll=10000.0,
        kelly_base=100.0,
        dlambert_config={"dlambert_enabled": True},
        rm=RM(),
        consecutive_losses_linear=8,
        pending_total=5000.0,
    )
    assert tag == "D'ALEMBERT"
    assert stake == pytest.approx(100.0 * (2.0**8))
    assert stake > 10000.0


def test_resolve_dlambert_stake_falls_back_to_kelly_when_disabled():
    class RM:
        dlambert_unit = 0.0
        dlambert_config = {}

    stake, tag = resolve_dlambert_stake(
        recovery_active=True,
        bankroll=10000.0,
        kelly_base=40.0,
        dlambert_config={"dlambert_enabled": False},
        rm=RM(),
        consecutive_losses_linear=4,
        pending_total=200.0,
    )
    assert tag == "KELLY"
    assert stake == pytest.approx(40.0)


def test_dlambert_log_suffix_geometric_and_empty():
    suffix = dlambert_log_suffix(
        "D'ALEMBERT",
        400.0,
        350.0,
        50.0,
        consecutive_losses_linear=3,
    )
    assert "2^3" in suffix
    assert "kelly=$50.00" in suffix
    assert dlambert_log_suffix("KELLY", 55.0, 0.0, 55.0) == ""
