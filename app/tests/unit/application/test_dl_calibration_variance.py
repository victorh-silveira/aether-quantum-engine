"""Guarda de variancia do calibrador → identity."""

from src.application.services.deep_learning.dl_calibration import CalibratorState
from src.application.services.deep_learning.dl_calibration_variance import (
    maybe_identity_on_variance_collapse,
)


def test_variance_collapse_switches_to_identity():
    preferred = CalibratorState(
        method="isotonic",
        isotonic_x=(0.4, 0.5, 0.6),
        isotonic_y=(0.5, 0.5, 0.5),
    )
    probs = [0.2, 0.35, 0.5, 0.65, 0.8]
    out = maybe_identity_on_variance_collapse(preferred, probs=probs)
    assert out.method == "identity"


def test_variance_guard_keeps_identity():
    preferred = CalibratorState(method="identity", temperature=1.0, platt_a=1.0, platt_b=0.0)
    probs = [0.2, 0.8]
    out = maybe_identity_on_variance_collapse(preferred, probs=probs)
    assert out.method == "identity"


def test_variance_guard_keeps_dispersed_calibrator():
    preferred = CalibratorState(method="temperature", temperature=1.0, platt_a=1.0, platt_b=0.0)
    probs = [0.1, 0.3, 0.5, 0.7, 0.9]
    out = maybe_identity_on_variance_collapse(preferred, probs=probs)
    assert out.method == "temperature"


def test_variance_guard_short_probs_keeps_preferred():
    preferred = CalibratorState(method="temperature", temperature=1.0, platt_a=1.0, platt_b=0.0)
    out = maybe_identity_on_variance_collapse(preferred, probs=[0.7])
    assert out.method == "temperature"


def test_variance_guard_empty_probs_keeps_preferred():
    preferred = CalibratorState(method="platt", temperature=1.0, platt_a=1.0, platt_b=0.0)
    out = maybe_identity_on_variance_collapse(preferred, probs=[])
    assert out.method == "platt"
