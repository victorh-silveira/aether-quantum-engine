"""Testes unitarios para o modulo de conformal prediction e quantificacao de incerteza."""

from src.application.services.deep_learning.dl_conformal_uncertainty import (
    ConformalPredictionBuffer,
    compute_conformal_quantile,
    evaluate_conformal_gate,
    get_global_conformal_buffer,
)


def test_compute_conformal_quantile_empty_and_bounds():
    assert compute_conformal_quantile([]) == 1.0
    assert compute_conformal_quantile([0.1, 0.2], alpha=0.01) == 1.0

    scores = [0.05, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30, 0.35]
    q = compute_conformal_quantile(scores, alpha=0.10)
    assert 0.0 < q <= 1.0


def test_conformal_buffer_record_and_clear():
    buf = ConformalPredictionBuffer(maxlen=64)
    assert buf.sample_count == 0
    buf.record_outcome(0.60, 1)
    buf.record_outcome(0.40, 0)
    buf.record_outcome("invalid", 1)
    assert buf.sample_count == 2
    buf.clear()
    assert buf.sample_count == 0


def test_conformal_buffer_evaluate_insufficient_samples():
    buf = ConformalPredictionBuffer(maxlen=64)
    buf.record_outcome(0.60, 1)
    res = buf.evaluate_uncertainty(0.58, min_samples=5)
    assert res["is_uncertain"] is False
    assert res["reason"] == "insufficient_samples"
    assert res["sample_size"] == 1


def test_conformal_buffer_diagnoses_uncertainty_when_spanning_neutral():
    buf = ConformalPredictionBuffer(maxlen=64)
    for _ in range(20):
        buf.record_outcome(0.60, 0)
        buf.record_outcome(0.40, 1)

    res = buf.evaluate_uncertainty(0.53, alpha=0.10, min_samples=10)
    assert res["is_uncertain"] is True
    assert res["reason"] == "conformal_uncertainty_excessive"
    assert res["p_lower"] <= 0.50 <= res["p_upper"]


def test_conformal_buffer_allows_when_sharp_outside_neutral():
    buf = ConformalPredictionBuffer(maxlen=64)
    for _ in range(20):
        buf.record_outcome(0.85, 1)
        buf.record_outcome(0.82, 1)

    res = buf.evaluate_uncertainty(0.80, alpha=0.10, min_samples=10)
    assert res["is_uncertain"] is False
    assert res["reason"] == "ok"
    assert res["p_lower"] > 0.50


def test_evaluate_conformal_gate_decorates_metrics():
    buf = ConformalPredictionBuffer(maxlen=64)
    for _ in range(15):
        buf.record_outcome(0.55, 0)

    metrics = {}
    vetoed, details = evaluate_conformal_gate(metrics, 0.52, buffer=buf, min_samples=5)
    assert vetoed is True
    assert "conformal_q" in metrics
    assert "conformal_p_lower" in metrics
    assert "conformal_p_upper" in metrics
    assert metrics["conformal_uncertainty_excessive"] is True

    vetoed_none, _ = evaluate_conformal_gate({}, None, buffer=buf)
    assert vetoed_none is False


def test_get_global_conformal_buffer():
    buf = get_global_conformal_buffer()
    assert isinstance(buf, ConformalPredictionBuffer)
