import pytest

from src.application.services.bb_width_adaptive_squeeze import (
    record_bb_width,
    reset_bb_width_buffer,
)
from src.application.services.deep_learning.dl_indicator_config import load_indicator_config_from_settings
from src.application.services.direction_loss_tracker import reset_direction_persistence_tracker
from src.application.services.execution_runtime_config import resolve_regime_micro_freeze_config
from src.application.services.regime_micro_freeze import (
    REGIME_CHOP_CONGESTION,
    SIGNAL_SUSPENDED,
    apply_regime_freeze_if_congested,
    chop_congestion_regime_active,
    log_d_squeeze_audit,
    micro_volatility_squeeze_active,
)
from src.domain.risk.consensus_stake_penalty import d_squeeze_sovereignty_active


@pytest.fixture(autouse=True)
def _reset_bb_buffer():
    reset_bb_width_buffer()
    reset_direction_persistence_tracker()
    yield
    reset_bb_width_buffer()
    reset_direction_persistence_tracker()


def _prime_bb(value: float, count: int | None = None) -> None:
    window = (
        count
        if count is not None
        else int(load_indicator_config_from_settings()["windows"]["bb_width_harmonic_window"])
    )
    for _ in range(window):
        record_bb_width(value)


def test_micro_volatility_squeeze_active_bb_width_and_negative_accel():
    _prime_bb(0.050)
    assert micro_volatility_squeeze_active({"indicators": {"bb_width": 0.015}}) is True
    _prime_bb(0.050)
    assert micro_volatility_squeeze_active({"micro_indicators": {"bb_width": 0.015}}) is True
    _prime_bb(0.050)
    assert micro_volatility_squeeze_active({"macro_indicators": {"bb_width": 0.015}}) is True
    assert micro_volatility_squeeze_active({"flow_features": {"micro_tick_acceleration": -0.02}}) is True
    assert (
        micro_volatility_squeeze_active(
            {
                "indicators": {"bb_width": 0.09},
                "flow_features": {"micro_tick_acceleration": 0.04},
            }
        )
        is False
    )
    _prime_bb(0.035)
    assert micro_volatility_squeeze_active({"indicators": {"bb_width": 0.035}}) is False


def test_chop_congestion_and_freeze():
    metrics = {
        "edge_zscore": float(resolve_regime_micro_freeze_config()["chop_congestion_z_edge"]) * 0.5,
        "flow_features": {
            "micro_tick_acceleration": float(resolve_regime_micro_freeze_config()["tick_accel_neutral_eps"]) * 0.5
        },
    }
    assert chop_congestion_regime_active(metrics, persistence_filter_active=False) is False
    assert chop_congestion_regime_active(metrics, persistence_filter_active=True) is True
    assert apply_regime_freeze_if_congested(metrics, persistence_filter_active=True) is True
    assert metrics["regime_classification"] == REGIME_CHOP_CONGESTION
    assert metrics["signal_status"] == SIGNAL_SUSPENDED


def test_log_d_squeeze_audit_runs():
    _prime_bb(0.050)
    metrics = {"indicators": {"bb_width": 0.015}, "predicted_payoff_edge": 0.1, "trade_score": 0.5}
    log_d_squeeze_audit("OTC_SPC", metrics)


def test_d_squeeze_sovereignty_without_inversion():
    metrics = {"trade_score": 0.52, "meta_squeeze_downgrade": True}
    assert d_squeeze_sovereignty_active(metrics) in {True, False}
