import pytest

from src.application.services.deep_learning.dl_calibration_tolerance import (
    infer_direction_from_prob as direction_from_raw_prob,
)
from src.application.services.deep_learning.dl_gating import (
    resolve_calibrated_edge,
    resolve_confidence_thresholds,
    resolve_edge,
)
from src.domain.models.trade import TradeDirection


def test_resolve_edge():
    actual = resolve_edge(0.80, horizon_bars=1)
    assert actual == pytest.approx(0.56, abs=0.01)


def test_resolve_edge_no_edge_at_fifty():
    assert resolve_edge(0.50) == 0.0


def test_resolve_edge_horizon_adjusts_payout():
    edge_1bar = resolve_edge(0.80, horizon_bars=1)
    edge_4bar = resolve_edge(0.80, horizon_bars=4)
    assert edge_4bar < edge_1bar


def test_resolve_calibrated_edge_prefers_calibrated():
    actual = resolve_calibrated_edge(0.82, raw_prob=0.60, horizon_bars=1)
    assert actual == pytest.approx(0.60, abs=0.01)


def test_resolve_calibrated_edge_falls_back_to_raw():
    assert resolve_calibrated_edge(None, raw_prob=0.70, horizon_bars=1) == pytest.approx(0.37, abs=0.01)


def test_resolve_calibrated_edge_defaults_to_zero():
    assert resolve_calibrated_edge(None) == 0.0


def test_confidence_thresholds_from_params():
    call_thr, put_thr = resolve_confidence_thresholds(
        {"confidence_call_threshold": 0.80, "confidence_put_threshold": 0.20}
    )
    assert call_thr == 0.80
    assert put_thr == 0.20


def test_direction_from_raw_prob_call():
    assert direction_from_raw_prob(0.80, None) == TradeDirection.CALL


def test_direction_from_raw_prob_put():
    assert direction_from_raw_prob(0.20, None) == TradeDirection.PUT


def test_direction_from_raw_prob_preserves_explicit():
    assert direction_from_raw_prob(0.50, TradeDirection.PUT) == TradeDirection.PUT


def test_resolve_edge_none_returns_zero():
    assert resolve_edge(None) == 0.0


def test_confidence_thresholds_non_dict_returns_default():
    call_thr, put_thr = resolve_confidence_thresholds(None)
    assert call_thr == 0.55
    assert put_thr == 0.45
