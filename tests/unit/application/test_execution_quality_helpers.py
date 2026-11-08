"""Cobertura de helpers de margem sem quality veto."""

import pytest

from src.application.services.execution_quality_gate import apply_quality_penalty_to_metrics, read_risk_session_state
from src.application.services.execution_quality_gate_margin import (
    direction_margin_from_probability,
    ensure_direction_margin,
)


def test_direction_margin_helpers():
    assert direction_margin_from_probability(0.70, direction="CALL") == pytest.approx(0.20)
    metrics = {"calibrated_prob": 0.70, "resolved_direction": "CALL"}
    assert ensure_direction_margin(metrics) == pytest.approx(0.20)
    assert apply_quality_penalty_to_metrics(metrics) == 0.0


def test_read_risk_session_state_defaults():
    linear, pending = read_risk_session_state(None)
    assert linear == 0
    assert pending == 0.0
