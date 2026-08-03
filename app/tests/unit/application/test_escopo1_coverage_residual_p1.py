"""Cobertura residual (parte 2) apos remocao dos vetos."""

from __future__ import annotations

import json
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.application.services.execution_quality_gate import read_risk_session_state
from src.application.services.execution_quality_gate_margin import stamp_edge_without_direction, sync_direction_margin
from src.application.services.execution_runtime_config import (
    reset_execution_runtime_cache,
    resolve_loss_protection_config,
    resolve_side_equilibrium_config,
)
from src.application.services.force_trade_mode import (
    force_trade_from_config,
    resolve_force_min_stake,
    synthesize_force_trade_candidate,
)
from src.application.services.infra_timing_config import (
    resolve_history_fetch_config,
    resolve_stream_reconnect_config,
    resolve_triton_infer_timeout,
)
from src.application.services.live_signal_metrics import _ece, apply_live_calib_drift_soft, record_live_signal_outcome
from src.application.services.live_signal_metrics_config import (
    reset_live_signal_metrics_config_cache,
    resolve_live_signal_metrics_config,
)
from src.domain.config_knobs import deep_merge, load_settings_json, merge_settings_block, require_keys, require_mapping


class _FakePath:
    def __init__(self, payload):
        self._payload = payload

    def open(self, *_a, **_k):
        text = self._payload if isinstance(self._payload, str) else json.dumps(self._payload)
        return StringIO(text)


def test_config_knobs_fail_closed_and_merge(monkeypatch):
    with pytest.raises(ValueError, match="obrigatorio"):
        require_mapping({}, "block", ("a",), "root")
    with pytest.raises(ValueError, match="incompleto"):
        require_mapping({"block": {}}, "block", ("a",), "root")
    with pytest.raises(ValueError, match="obrigatorio"):
        require_keys(None, ("a",), "path")
    merged = deep_merge({"a": {"b": 1, "c": 0}}, {"a": {"c": 2, "d": 3}, "e": 1})
    assert merged["a"]["b"] == 1
    assert merged["a"]["c"] == 2
    monkeypatch.setattr(
        "src.domain.config_knobs.repo_path",
        lambda *a, **k: _FakePath("[1, 2]"),
    )
    with pytest.raises(ValueError, match="invalido"):
        load_settings_json()
    monkeypatch.setattr(
        "src.domain.config_knobs.repo_path",
        lambda *a, **k: _FakePath({"orchestrator": {}}),
    )
    with pytest.raises(ValueError, match="execution"):
        merge_settings_block(("orchestrator", "execution"), None)


def test_execution_runtime_config_cache_merge_and_side_eq(monkeypatch):
    reset_execution_runtime_cache()
    side = resolve_side_equilibrium_config(None)
    assert side["small_window"] >= 1
    lp = resolve_loss_protection_config({"loss_protection": {"min_direction_margin": 0.19}})
    assert lp["min_direction_margin"] == pytest.approx(0.19)
    reset_execution_runtime_cache()
    monkeypatch.setattr(
        "src.application.services.execution_runtime_config.repo_path",
        lambda *a, **k: _FakePath({"orchestrator": {"execution": "x"}}),
    )
    reset_execution_runtime_cache()
    with pytest.raises(ValueError, match="execution"):
        resolve_side_equilibrium_config(None)
    reset_execution_runtime_cache()


def test_execution_runtime_config_bad_loss_protection_disconnect():
    reset_execution_runtime_cache()
    bad_disconnect = {
        "min_direction_margin": 0.1,
        "recovery_min_direction_margin": 0.1,
        "recovery_min_hurst": 0.4,
        "max_edge_without_margin": 0.1,
        "max_zscore_without_margin": 0.1,
        "disconnect": "not-a-dict",
    }
    with pytest.raises(ValueError, match="disconnect"):
        resolve_loss_protection_config({"loss_protection": bad_disconnect})


def test_infra_timing_config_branches(monkeypatch):
    monkeypatch.setattr(
        "src.application.services.infra_timing_config.repo_path",
        lambda *a, **k: _FakePath([]),
    )
    with pytest.raises(ValueError, match="invalido"):
        from src.application.services.infra_timing_config import _load_settings

        _load_settings()
    reset_execution_runtime_cache()
    flat = resolve_stream_reconnect_config(
        {"max_attempts": 9, "initial_backoff_seconds": 1.0, "max_backoff_seconds": 2.0}
    )
    assert flat["max_attempts"] == 9
    hist = resolve_history_fetch_config({"chunk": 500, "delay_seconds": 0.1})
    assert hist["chunk"] == 500
    with pytest.raises(ValueError, match="infer_timeout"):
        resolve_triton_infer_timeout({})


def test_live_signal_metrics_config_paths():
    reset_live_signal_metrics_config_cache()
    base = resolve_live_signal_metrics_config(None)
    override = resolve_live_signal_metrics_config({"live_signal_metrics": dict(base)})
    assert override["window"] == base["window"]
    reset_live_signal_metrics_config_cache()


def test_live_signal_metrics_drift_and_ece():
    assert _ece([], []) == pytest.approx(0.0)
    orch = SimpleNamespace(_live_signal_metrics=None, _active_cycle_id=2)
    metrics = {"live_n": 40, "live_ece": 0.4, "live_wr": 0.1, "raw_prob": 0.9, "trade_score": 0.7}
    with (
        patch(
            "src.application.services.live_signal_metrics.load_sample_size_policy",
            return_value={"calib_soft_min_n": 1},
        ),
        patch(
            "src.application.services.live_signal_metrics._live",
            return_value={
                "ece_soft_threshold": 0.05,
                "drift_soft_penalty": 0.1,
                "drift_soft_veto_n": 10,
                "drift_min_score": 0.4,
                "drift_score_factor": 0.5,
                "window": 64,
                "min_rank": 8,
                "ece_bins": 10,
            },
        ),
    ):
        assert apply_live_calib_drift_soft(metrics, orch=orch, symbol="R_10") is True
        assert metrics.get("calib_drift_soft_veto") is True
    for i in range(8):
        record_live_signal_outcome(orch, "R_10", won=i % 2 == 0, raw_prob=0.6, direction="CALL")


def test_force_trade_and_quality_margin_branches():
    assert force_trade_from_config({"orchestrator": "x"}) is False
    assert resolve_force_min_stake({"risk_management": "x"}) > 0.0
    entry = {
        "metrics": {
            "deploy_ok": True,
            "raw_prob": 0.7,
            "gate_reason": "neutral_clamp",
            "calibration_mode": "neutral_clamp",
        }
    }
    picked = synthesize_force_trade_candidate(["R_10"], {"R_10": entry}, orch=SimpleNamespace())
    assert picked is not None
    assert picked[2]["force_trade_every_cycle"] is True
    metrics = {"trade_score": 0.8, "predicted_payoff_edge": 0.2}
    stamp_edge_without_direction(metrics, margin_floor=0.1, score_factor=0.5)
    assert metrics["edge_without_direction_penalty"] > 0.0
    sync_metrics = {"direction_call_score": 0.6, "direction_put_score": 0.4}
    sync_direction_margin(sync_metrics, direction="CALL")
    assert sync_metrics["direction_margin"] == pytest.approx(0.2)
    rm = SimpleNamespace(pending_loss={"R_10": 1.0})
    _, pending = read_risk_session_state(rm)
    assert pending == pytest.approx(1.0)
