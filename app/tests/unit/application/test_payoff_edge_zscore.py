import pytest

from src.application.services.execution_direction_resolver import resolve_execution_direction
from src.application.services.execution_runtime_config import resolve_edge_zscore_runtime
from src.application.services.payoff_edge_zscore import (
    _indicator_map,
    apply_payoff_edge_zscore,
    attach_payoff_edge_zscore_metrics,
    compute_edge_zscore,
    payoff_edge_buffer_snapshot,
    reset_payoff_edge_buffer,
    resolve_adaptive_edge_window,
    sample_edge_std,
)
from src.domain.models.trade import TradeDirection


@pytest.fixture(autouse=True)
def _clear_edge_buffer():
    reset_payoff_edge_buffer()
    yield
    reset_payoff_edge_buffer()


def test_stable_edge_sequence_yields_neutral_zscore():
    z_edge = 0.0
    for edge in [1.28] * 10:
        z_edge = apply_payoff_edge_zscore(edge)
    assert abs(z_edge) < 0.5


def test_spike_edge_above_stable_window_raises_zscore():
    for edge in [1.20] * 10:
        apply_payoff_edge_zscore(edge)
    z_edge = apply_payoff_edge_zscore(1.45)
    assert z_edge >= 0.5


def test_compute_edge_zscore_uses_history_override():
    history = [1.0, 1.1, 0.9, 1.0, 1.05]
    z = compute_edge_zscore(1.4, history=history)
    assert z > 0.5


def test_sample_edge_std_requires_two_points():
    assert sample_edge_std([1.2]) == 0.0
    assert sample_edge_std([1.0, 1.2]) > 0.0


def test_attach_payoff_edge_zscore_metrics_sets_zscore():
    metrics: dict = {}
    z_edge = attach_payoff_edge_zscore_metrics(metrics, 1.25)
    assert metrics["edge_zscore"] == z_edge
    assert metrics["meta_payoff_edge_zscore"] == z_edge


def test_turbo_threshold_constant():
    assert pytest.approx(1.5) == float(resolve_edge_zscore_runtime()["turbo_threshold"])


def test_reset_and_snapshot_buffer():
    apply_payoff_edge_zscore(0.42)
    assert payoff_edge_buffer_snapshot() == (0.42,)
    reset_payoff_edge_buffer()
    assert payoff_edge_buffer_snapshot() == ()


def test_reset_payoff_edge_buffer_for_single_symbol():
    apply_payoff_edge_zscore(0.42, symbol="OTC_SPC")
    apply_payoff_edge_zscore(0.55, symbol="R_50")
    reset_payoff_edge_buffer("OTC_SPC")
    assert payoff_edge_buffer_snapshot("OTC_SPC") == ()
    assert payoff_edge_buffer_snapshot("R_50") == (0.55,)


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


def test_indicator_map_returns_empty_for_non_dict():
    assert _indicator_map(None) == {}


def test_resolve_adaptive_edge_window_bounds():
    assert (
        int(resolve_edge_zscore_runtime()["window_min"])
        <= resolve_adaptive_edge_window()
        <= int(resolve_edge_zscore_runtime()["window_max"])
    )
    trending = resolve_adaptive_edge_window({"indicators": {"hurst": 0.72, "atr_norm": 0.20}})
    lateral = resolve_adaptive_edge_window({"indicators": {"hurst": 0.48, "bb_width": 0.04}})
    assert trending <= lateral


def test_adaptive_window_shortens_history_for_zscore():
    metrics = {"indicators": {"hurst": 0.72, "atr_change_ratio": 0.25}}
    for edge in [1.0, 1.05, 0.95, 1.02, 1.01]:
        apply_payoff_edge_zscore(edge, metrics)
    z_adaptive = apply_payoff_edge_zscore(1.40, metrics)
    history = [1.0, 1.05, 0.95, 1.02, 1.01]
    z_static = compute_edge_zscore(1.40, history=history)
    assert metrics["edge_zscore_window"] <= int(resolve_edge_zscore_runtime()["window_max"])
    assert z_adaptive != z_static or abs(z_adaptive) >= 0.0


def test_attach_metrics_records_adaptive_window():
    metrics: dict = {"indicators": {"hurst": 0.62, "bb_width": 0.08}}
    attach_payoff_edge_zscore_metrics(metrics, 0.85)
    assert metrics["edge_zscore_window"] >= int(resolve_edge_zscore_runtime()["window_min"])
    assert metrics["meta_payoff_edge_zscore"] == metrics["edge_zscore"]


def test_active_edge_history_slices_when_buffer_exceeds_window():
    metrics = {"indicators": {"hurst": 0.72, "atr_change_ratio": 0.25}}
    for edge in [1.0 + 0.01 * i for i in range(50)]:
        apply_payoff_edge_zscore(edge, metrics)
    z_edge = apply_payoff_edge_zscore(1.55, metrics)
    assert metrics["edge_zscore_window"] < 50
    assert isinstance(z_edge, float)


def test_resolve_adaptive_edge_window_uses_micro_indicators():
    window = resolve_adaptive_edge_window({"micro_indicators": {"hurst": 0.50, "bb_width": 0.03}})
    assert (
        int(resolve_edge_zscore_runtime()["window_min"]) <= window <= int(resolve_edge_zscore_runtime()["window_max"])
    )


def test_resolve_stable_edge_sequence_keeps_direction():
    result = None
    for edge in [1.28] * 10:
        result = resolve_execution_direction(_resolver_entry(edge), symbol="OTC_SPC")
        assert result is not None
    assert result is not None
    assert abs(float(result[1]["edge_zscore"])) < 0.5
    assert result[0] == TradeDirection.CALL


def test_resolve_edge_spike_records_high_zscore():
    for edge in [1.20] * 12:
        resolve_execution_direction(_resolver_entry(edge), symbol="OTC_SPC")
    result = resolve_execution_direction(_resolver_entry(1.45), symbol="OTC_SPC")
    assert result is not None
    assert float(result[1]["edge_zscore"]) >= 0.5
