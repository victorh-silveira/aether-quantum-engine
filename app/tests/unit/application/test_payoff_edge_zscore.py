import pytest

from src.application.services.execution_direction_resolver import resolve_execution_direction
from src.application.services.payoff_edge_zscore import (
    EDGE_ZSCORE_TURBO_THRESHOLD,
    LOSS_EXPECTED,
    NO_EDGE_NEUTRAL,
    WIN_EXPECTED,
    apply_payoff_edge_zscore,
    attach_payoff_edge_zscore_metrics,
    classify_edge_expectancy,
    compute_edge_zscore,
    edge_zscore_neutral_regime_active,
    payoff_edge_buffer_snapshot,
    reset_payoff_edge_buffer,
    sample_edge_std,
)
from src.domain.models.trade import TradeDirection


@pytest.fixture(autouse=True)
def _clear_edge_buffer():
    reset_payoff_edge_buffer()
    yield
    reset_payoff_edge_buffer()


def test_classify_edge_expectancy_states():
    assert classify_edge_expectancy(0.2, 1.0) == WIN_EXPECTED
    assert classify_edge_expectancy(0.2, 0.49) == NO_EDGE_NEUTRAL
    assert classify_edge_expectancy(0.2, -0.2) == NO_EDGE_NEUTRAL
    assert classify_edge_expectancy(-0.1, 2.0) == LOSS_EXPECTED


def test_stable_edge_sequence_yields_neutral_zscore():
    z_edge = 0.0
    state = NO_EDGE_NEUTRAL
    for edge in [1.28] * 10:
        z_edge, state = apply_payoff_edge_zscore(edge)
    assert abs(z_edge) < 0.5
    assert state == NO_EDGE_NEUTRAL


def test_spike_edge_above_stable_window_flags_win():
    for edge in [1.20] * 10:
        apply_payoff_edge_zscore(edge)
    z_edge, state = apply_payoff_edge_zscore(1.45)
    assert z_edge >= 0.5
    assert state == WIN_EXPECTED


def test_compute_edge_zscore_uses_history_override():
    history = [1.0, 1.1, 0.9, 1.0, 1.05]
    z = compute_edge_zscore(1.4, history=history)
    assert z > 0.5


def test_sample_edge_std_requires_two_points():
    assert sample_edge_std([1.2]) == 0.0
    assert sample_edge_std([1.0, 1.2]) > 0.0


def test_attach_payoff_edge_zscore_metrics_sets_regime_flag():
    metrics: dict = {}
    attach_payoff_edge_zscore_metrics(metrics, 1.25)
    assert "edge_zscore" in metrics
    assert metrics["edge_expectancy"] in {WIN_EXPECTED, NO_EDGE_NEUTRAL, LOSS_EXPECTED}
    assert isinstance(metrics["edge_neutral_regime"], bool)


def test_edge_zscore_neutral_regime_active():
    assert edge_zscore_neutral_regime_active({"edge_expectancy": NO_EDGE_NEUTRAL}) is True
    assert edge_zscore_neutral_regime_active({"edge_expectancy": WIN_EXPECTED, "edge_zscore": 0.8}) is False


def test_turbo_threshold_constant():
    assert pytest.approx(1.5) == EDGE_ZSCORE_TURBO_THRESHOLD


def test_reset_and_snapshot_buffer():
    apply_payoff_edge_zscore(0.42)
    assert payoff_edge_buffer_snapshot() == (0.42,)
    reset_payoff_edge_buffer()
    assert payoff_edge_buffer_snapshot() == ()


def test_compute_edge_zscore_returns_zero_with_short_history():
    assert compute_edge_zscore(1.0, history=[1.0]) == 0.0


def _resolver_entry(edge: float) -> dict:
    return {
        "direction": TradeDirection.CALL,
        "metrics": {
            "deploy_ok": True,
            "calibrated_prob": 0.70,
            "predicted_payoff_edge": edge,
            "meta_classifier_applied": True,
        },
    }


def test_resolve_stable_edge_sequence_marks_no_edge_neutral():
    result = None
    for edge in [1.28] * 10:
        result = resolve_execution_direction(_resolver_entry(edge), symbol="RDBULL")
        assert result is not None
    assert result is not None
    assert result[1]["edge_expectancy"] == NO_EDGE_NEUTRAL
    assert abs(float(result[1]["edge_zscore"])) < 0.5


def test_resolve_edge_spike_above_window_marks_win_expected():
    for edge in [1.20] * 12:
        resolve_execution_direction(_resolver_entry(edge), symbol="RDBULL")
    result = resolve_execution_direction(_resolver_entry(1.45), symbol="RDBULL")
    assert result is not None
    assert result[1]["edge_expectancy"] == WIN_EXPECTED
    assert float(result[1]["edge_zscore"]) >= 0.5
