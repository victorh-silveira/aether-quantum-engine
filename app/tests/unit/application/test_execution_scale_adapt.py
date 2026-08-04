"""Testes de adaptacao de direcao e teto de stake SCALE."""

from unittest.mock import patch

from src.application.services.execution_scale_adapt import apply_scale_direction_adapt
from src.application.services.execution_scale_sizing import apply_scale_kelly_sizing
from src.domain.models.trade import TradeDirection
from src.domain.risk.dlambert_sizing import resolve_dlambert_stake
from src.domain.risk.risk_stake_calc_helpers import apply_scale_stake_cap


def test_adapt_direction_tape_vs_tcn_raw_extreme():
    metrics = {
        "scale_tape_consensus": "CALL",
        "calibration_mode": "raw_extreme",
    }
    out = apply_scale_direction_adapt(metrics, TradeDirection.PUT)
    assert out == TradeDirection.CALL
    assert metrics["scale_adapted"] is True
    assert metrics["tcn_direction"] == "PUT"
    assert metrics["scale_adapt_reason"] == "tape_vs_tcn"


def test_adapt_requires_raw_extreme():
    metrics = {"scale_tape_consensus": "CALL", "calibration_mode": "calibrated"}
    out = apply_scale_direction_adapt(metrics, TradeDirection.PUT)
    assert out == TradeDirection.PUT
    assert metrics["scale_adapted"] is False
    assert metrics["scale_adapt_reason"] == "need_raw_extreme"


def test_adapt_no_consensus_keeps_tcn():
    metrics = {"scale_tape_consensus": None, "calibration_mode": "raw_extreme"}
    out = apply_scale_direction_adapt(metrics, TradeDirection.PUT)
    assert out == TradeDirection.PUT
    assert metrics["scale_adapt_reason"] == "no_consensus"


def test_adapt_aligned_consensus():
    metrics = {"scale_tape_consensus": "PUT", "calibration_mode": "raw_extreme"}
    out = apply_scale_direction_adapt(metrics, TradeDirection.PUT)
    assert out == TradeDirection.PUT
    assert metrics["scale_adapt_reason"] == "aligned"


def test_adapt_disabled_flags():
    metrics = {"scale_tape_consensus": "CALL", "calibration_mode": "raw_extreme"}
    with patch(
        "src.application.services.execution_scale_adapt.parse_scale_vision_config",
        return_value={"enabled": False, "adapt_direction_enabled": True, "adapt_require_raw_extreme": True},
    ):
        assert apply_scale_direction_adapt(metrics, TradeDirection.PUT) == TradeDirection.PUT
    metrics2 = {"scale_tape_consensus": "CALL", "calibration_mode": "raw_extreme"}
    with patch(
        "src.application.services.execution_scale_adapt.parse_scale_vision_config",
        return_value={"enabled": True, "adapt_direction_enabled": False, "adapt_require_raw_extreme": True},
    ):
        assert apply_scale_direction_adapt(metrics2, TradeDirection.PUT) == TradeDirection.PUT
        assert metrics2["scale_adapt_reason"] == "adapt_off"


def test_sizing_on_adapted_sets_force_explore_and_cap():
    metrics = {"kelly_fraction_scale": 1.0, "scale_adapted": True, "scale_discordance": False}
    apply_scale_kelly_sizing(None, "R_10", TradeDirection.CALL, metrics)
    assert metrics["scale_force_explore"] is True
    assert metrics["kelly_fraction_scale"] < 1.0
    assert float(metrics["scale_max_stake_pct"]) > 0.0


def test_stake_cap_discord_and_noop():
    assert apply_scale_stake_cap(100.0, 10000.0, None) == 100.0
    assert apply_scale_stake_cap(100.0, 10000.0, {"scale_discordance": False}) == 100.0
    assert apply_scale_stake_cap(100.0, 10000.0, {"scale_adapted": True, "scale_max_stake_pct": None}) == 100.0
    capped = apply_scale_stake_cap(100.0, 10000.0, {"scale_adapted": True, "scale_max_stake_pct": 0.005})
    assert capped == 50.0
    assert apply_scale_stake_cap(10.0, 10000.0, {"scale_adapted": True, "scale_max_stake_pct": "bad"}) == 10.0


def test_resolve_dlambert_skips_dal_on_scale_adapted():
    class RM:
        soft_recovery_config = {"enabled": True, "max_safe_stake_cap": 300.0, "max_safe_stake_pct": 0.03}
        last_loss_stake = 50.0
        dlambert_unit = 1.0
        risk_params = {}
        total_session_profit = 0.0
        daily_stop_win_target = 0.0

    stake, tag = resolve_dlambert_stake(
        recovery_active=True,
        bankroll=5000.0,
        kelly_base=20.0,
        dlambert_config={"dlambert_enabled": True},
        rm=RM(),
        consecutive_losses_linear=3,
        pending_total=80.0,
        payout=0.87,
        dl_metrics={"scale_adapted": True, "execute": True},
        f_star=0.01,
    )
    assert tag == "KELLY"
    assert stake > 0.0


def test_resolve_dlambert_skips_dal_on_scale_force_explore():
    class RM:
        soft_recovery_config = {"enabled": True, "max_safe_stake_cap": 300.0, "max_safe_stake_pct": 0.03}
        last_loss_stake = 50.0
        dlambert_unit = 1.0
        risk_params = {}
        total_session_profit = 0.0
        daily_stop_win_target = 0.0

    stake, tag = resolve_dlambert_stake(
        recovery_active=True,
        bankroll=5000.0,
        kelly_base=20.0,
        dlambert_config={"dlambert_enabled": True},
        rm=RM(),
        consecutive_losses_linear=3,
        pending_total=80.0,
        payout=0.87,
        dl_metrics={"scale_force_explore": True, "execute": True},
        f_star=0.01,
    )
    assert tag == "KELLY"
    assert stake > 0.0


def test_finalize_adapts_direction_under_raw_extreme():
    import numpy as np

    from src.application.services.execution_direction_resolver import _finalize_execution_metrics

    entry = {"metrics": {}}
    metrics = {
        "calibration_mode": "raw_extreme",
        "flow_features": {"price_velocity": 1.0},
        "kelly_fraction_scale": 1.0,
    }

    class Stream:
        def get_numpy_series(self, _symbol, _field="close"):
            return np.array([1.0, 1.05, 1.1, 1.15, 1.2])

        def get_mini_numpy_series(self, _symbol, field="close"):
            if field == "open":
                return np.array([1.0, 1.0, 1.0, 1.0, 1.0])
            return np.array([1.0, 1.05, 1.1, 1.15, 1.2])

        def get_micro_numpy_series(self, _symbol, field="close"):
            if field == "open":
                return np.array([1.0, 1.0, 1.0])
            return np.array([0.95, 1.05, 1.15])

        tick_buffer = None

    orch = type("O", (), {"stream": Stream()})()
    direction, out = _finalize_execution_metrics(
        entry,
        metrics,
        TradeDirection.PUT,
        0.2,
        0.01,
        meta_applied=False,
        score=0.55,
        symbol="R_10",
        orch=orch,
    )
    assert direction == TradeDirection.CALL
    assert out["scale_adapted"] is True
    assert out["tcn_direction"] == "PUT"
    assert out["execution_candidate_ready"] is True
    assert out["scale_force_explore"] is True
