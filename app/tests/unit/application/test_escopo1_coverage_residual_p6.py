"""Cobertura residual (parte 2) apos remocao dos vetos."""

from __future__ import annotations

import json
import logging
from io import StringIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.application.services.execution_direction_fallback import build_mandatory_fallback_candidate
from src.application.services.execution_market_rank import market_decision_score
from src.application.services.execution_quality_gate import read_risk_session_state
from src.application.services.execution_runtime_config import (
    reset_execution_runtime_cache,
    resolve_loss_protection_config,
)
from src.application.services.force_trade_mode import (
    synthesize_force_trade_candidate,
)
from src.application.services.infra_timing_config import (
    resolve_history_fetch_config,
    resolve_stream_reconnect_config,
)
from src.application.services.live_signal_metrics import apply_live_calib_drift_soft
from src.application.services.live_signal_metrics_config import (
    reset_live_signal_metrics_config_cache,
)
from src.application.services.orchestrator.execution_blockers import log_execution_blockers
from src.domain.models.trade import TradeDirection


class _FakePath:
    def __init__(self, payload):
        self._payload = payload

    def open(self, *_a, **_k):
        text = self._payload if isinstance(self._payload, str) else json.dumps(self._payload)
        return StringIO(text)


def test_residual_gaps_execution_quality_rank_force_fallback():
    linear, pending = read_risk_session_state(None, pending_loss_total=2.5)
    assert pending == pytest.approx(2.5)
    assert synthesize_force_trade_candidate(["OTC_SPC"], "not-dict") is None
    metrics_soft = {
        "live_n": 50,
        "live_brier": 0.35,
        "live_ece": 0.5,
        "val_ece": 0.1,
        "trade_score": 0.6,
        "edge": 0.1,
        "execute": True,
        "deploy_ok": True,
        "direction_margin": 0.2,
        "indicators": {},
    }
    assert market_decision_score(metrics_soft) != market_decision_score({**metrics_soft, "live_brier": 0.1})
    orch = SimpleNamespace(_active_cycle_id=1, config={})
    decisions = {
        "OTC_SPC": {
            "direction": TradeDirection.CALL,
            "metrics": {"deploy_ok": True, "raw_prob": 0.7, "trade_score": 0.8, "val_accuracy": 0.6},
        }
    }
    last = ("OTC_SPC", TradeDirection.CALL, decisions["OTC_SPC"]["metrics"])
    with (
        patch(
            "src.application.services.execution_direction_fallback.pick_best_mandatory_candidate",
            return_value=None,
        ),
        patch(
            "src.application.services.execution_direction_fallback._scored_fallback_pick",
            return_value=last,
        ),
    ):
        picked = build_mandatory_fallback_candidate(
            ["OTC_SPC"],
            decisions,
            recovery_active=False,
            last_loss_symbol=None,
            orch=orch,
        )
    assert picked == last


def test_execution_runtime_block_threshold_and_infra_nested(monkeypatch):
    reset_execution_runtime_cache()
    import src.application.services.execution_runtime_config as erc

    base = resolve_loss_protection_config(None)
    disconnect = {k: v for k, v in base["disconnect"].items() if k != "block_threshold"}
    bad_lp = {
        "min_direction_margin": base["min_direction_margin"],
        "recovery_min_direction_margin": base["recovery_min_direction_margin"],
        "recovery_min_hurst": base["recovery_min_hurst"],
        "max_edge_without_margin": base["max_edge_without_margin"],
        "max_zscore_without_margin": base["max_zscore_without_margin"],
        "disconnect": disconnect,
    }
    monkeypatch.setattr(
        erc,
        "_load_execution_from_settings",
        lambda: {"loss_protection": bad_lp},
    )
    with pytest.raises(ValueError, match="block_threshold"):
        resolve_loss_protection_config(None)
    reset_execution_runtime_cache()
    flat = resolve_stream_reconnect_config(
        {"max_attempts": 9, "initial_backoff_seconds": 1.0, "max_backoff_seconds": 2.0},
    )
    assert flat["max_attempts"] == 9
    flat_hist = resolve_history_fetch_config(
        {
            "chunk": 500,
            "delay_seconds": 0.1,
            "symbol_delay_seconds": 0.0,
            "rate_limit_retries": 1,
            "rate_limit_backoff": 1.0,
            "rate_limit_max_delay": 1.0,
        },
    )
    assert flat_hist["chunk"] == 500


def test_live_signal_config_missing_block(monkeypatch):
    reset_live_signal_metrics_config_cache()
    monkeypatch.setattr(
        "src.application.services.live_signal_metrics_config.repo_path",
        lambda *a, **k: _FakePath({"deep_learning": {}}),
    )
    reset_live_signal_metrics_config_cache()
    from src.application.services.live_signal_metrics_config import load_live_signal_metrics_from_settings

    with pytest.raises(ValueError, match="live_signal_metrics"):
        load_live_signal_metrics_from_settings()


def test_live_signal_drift_resolved_conviction_and_raw_side():
    orch = SimpleNamespace(_active_cycle_id=1)
    metrics = {"live_n": 20, "live_ece": 0.5, "live_wr": 0.1, "raw_prob": 0.9}
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
                "drift_soft_veto_n": 5,
                "drift_min_score": 0.4,
                "drift_score_factor": 0.5,
                "window": 64,
                "min_rank": 8,
                "ece_bins": 10,
            },
        ),
    ):
        apply_live_calib_drift_soft(metrics, orch=orch, symbol="OTC_SPC")
    metrics2 = {"live_n": 20, "live_ece": 0.5, "live_wr": 0.1, "raw_prob": 0.9, "resolved_conviction": 0.75}
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
                "drift_soft_veto_n": 5,
                "drift_min_score": 0.4,
                "drift_score_factor": 0.5,
                "window": 64,
                "min_rank": 8,
                "ece_bins": 10,
            },
        ),
    ):
        apply_live_calib_drift_soft(metrics2, orch=orch, symbol="OTC_SPC")


def test_execution_blockers_deploy_and_signal_suspended():
    executor = MagicMock()
    executor.orch = SimpleNamespace(
        _active_cycle_id=5,
        config={"orchestrator": {"execution": {}}},
    )
    executor._trade_symbols = MagicMock(return_value=["OTC_SPC", "R_50"])
    executor.logger = logging.getLogger("test.blockers")
    log_execution_blockers(
        executor,
        {
            "OTC_SPC": {"metrics": {"deploy_ok": False}},
            "R_50": {"metrics": {"signal_status": "SIGNAL_SUSPENDED"}},
        },
    )
