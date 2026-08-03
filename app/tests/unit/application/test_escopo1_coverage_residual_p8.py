"""Cobertura residual (parte 2) apos remocao dos vetos."""

from __future__ import annotations

import json
from io import StringIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.execution_market_rank import market_decision_score
from src.application.services.infra_timing_config import (
    resolve_history_fetch_config,
    resolve_stream_reconnect_config,
)
from src.application.services.live_signal_metrics import apply_live_calib_drift_soft
from src.application.services.meta_payoff_shadow import (
    meta_inverted_shadow_active,
    record_meta_payoff_shadow_pair,
    reset_meta_payoff_shadow,
)
from src.application.services.orchestrator.orchestrator_run_loop import (
    align_exec_empty_recovery_signature_cooldown,
)
from src.application.services.orchestrator.post_settlement_cycle import _await_exec_empty_signature_alignment
from src.application.services.regime_micro_freeze import apply_regime_freeze_if_congested


class _FakePath:
    def __init__(self, payload):
        self._payload = payload

    def open(self, *_a, **_k):
        text = self._payload if isinstance(self._payload, str) else json.dumps(self._payload)
        return StringIO(text)


def test_market_rank_soft_brier_and_ece_penalties():
    cfg = {
        "live_n_min": 10,
        "live_brier_hard_above": 0.5,
        "live_brier_soft_above": 0.3,
        "live_brier_hard_penalty": -0.2,
        "live_brier_soft_penalty": -0.05,
        "ece_hard_above": 0.4,
        "ece_soft_above": 0.2,
        "ece_hard_penalty": -0.15,
        "ece_soft_penalty": -0.04,
        "weight_trade_score": 1.0,
        "weight_edge": 0.0,
        "weight_margin": 0.0,
        "weight_meta_z": 0.0,
        "blend_primary": 1.0,
        "blend_secondary": 0.0,
        "execute_bonus": 0.0,
        "deploy_ok_bonus": 0.0,
        "last_loss_penalty": 0.0,
        "recovery_last_loss_penalty": 0.0,
        "rotate_bonus": 0.0,
        "adx_weak_penalty": 0.0,
        "adx_strong_bonus": 0.0,
        "hurst_trend_bonus": 0.0,
        "hurst_mean_revert_penalty": 0.0,
        "thin_margin_penalty": 0.0,
        "thin_margin_below": 0.0,
        "squeeze_recovery_penalty": 0.0,
        "indicator_defaults": {"adx": 0.3, "vol_ratio": 1.0, "hurst": 0.5},
    }
    metrics = {
        "live_n": 5,
        "val_brier": 0.35,
        "val_ece": 0.25,
        "trade_score": 0.6,
        "edge": 0.1,
        "execute": True,
        "deploy_ok": True,
        "direction_margin": 0.2,
        "indicators": {},
    }
    with patch("src.application.services.execution_market_rank._composite", return_value=cfg):
        soft_score = market_decision_score(metrics)
        hard_metrics = {**metrics, "val_brier": 0.55, "val_ece": 0.45}
        hard_score = market_decision_score(hard_metrics)
    assert soft_score > hard_score


def test_live_calib_drift_returns_false_when_consistent():
    metrics = {"live_n": 20, "live_ece": 0.02, "live_wr": 0.61, "raw_prob": 0.61}
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
        assert apply_live_calib_drift_soft(metrics) is False


def test_align_exec_empty_skips_zero_pending():
    orch = SimpleNamespace(
        _last_cycle_cluster_executed=False,
        config={"orchestrator": {}},
        risk_manager=SimpleNamespace(pending_loss_total=lambda: 0.0),
    )
    assert align_exec_empty_recovery_signature_cooldown(orch) == 0.0


def test_side_equilibrium_async_hset_and_timescale_task():
    async def _async_result():
        return None

    client = MagicMock()
    client.hset = MagicMock(return_value=_async_result())
    writer = MagicMock()
    writer.enqueue_trade_outcome = MagicMock(return_value=_async_result())
    orch = SimpleNamespace(
        config={"orchestrator": {"execution": {}}},
        state_store=SimpleNamespace(client=client),
        timescale_writer=writer,
        create_task=MagicMock(),
    )
    from src.application.services.side_equilibrium_store import record_side_equilibrium_outcome

    record_side_equilibrium_outcome(orch, "R_10", direction="CALL", won=True)
    orch.create_task.assert_called_once()


def test_infra_timing_flat_api_overrides():
    stream = resolve_stream_reconnect_config(
        {"stream_reconnect": {"max_attempts": 6, "initial_backoff_seconds": 1.0, "max_backoff_seconds": 3.0}},
    )
    assert stream["max_attempts"] == 6
    hist = resolve_history_fetch_config(
        {
            "history_fetch": {
                "chunk": 450,
                "delay_seconds": 0.2,
                "symbol_delay_seconds": 0.0,
                "rate_limit_retries": 1,
                "rate_limit_backoff": 1.0,
                "rate_limit_max_delay": 1.0,
            }
        },
    )
    assert hist["chunk"] == 450


def test_meta_inverted_shadow_false_when_corr_positive():
    reset_meta_payoff_shadow()
    for i in range(16):
        record_meta_payoff_shadow_pair(z_score=float(i), profit=float(i), orch=None)
    assert meta_inverted_shadow_active(None) is False
    reset_meta_payoff_shadow()
    record_meta_payoff_shadow_pair(z_score=1.0, profit=-1.0, orch=None)
    assert meta_inverted_shadow_active(None) is False


def test_align_exec_empty_skips_when_cluster_executed():
    orch = SimpleNamespace(
        _last_cycle_cluster_executed=True,
        config={"orchestrator": {}},
        risk_manager=SimpleNamespace(pending_loss_total=lambda: 10.0),
    )
    assert align_exec_empty_recovery_signature_cooldown(orch) == 0.0


@pytest.mark.asyncio
async def test_post_settlement_exec_empty_poll_when_no_delay():
    orch = SimpleNamespace(
        risk_manager=SimpleNamespace(pending_loss_total=lambda: 5.0),
        config={"orchestrator": {}},
        _cooldown_until=0.0,
    )
    with (
        patch(
            "src.application.services.orchestrator.post_settlement_cycle.seconds_until_next_signature_boundary",
            return_value=0.0,
        ),
        patch(
            "src.application.services.orchestrator.post_settlement_cycle._poll_delay",
            new_callable=AsyncMock,
        ) as poll,
    ):
        await _await_exec_empty_signature_alignment(orch, 0.01)
        poll.assert_awaited()


def test_regime_micro_freeze_inactive_returns_false():
    with patch(
        "src.application.services.regime_micro_freeze.chop_congestion_regime_active",
        return_value=False,
    ):
        metrics = {}
        assert apply_regime_freeze_if_congested(metrics, persistence_filter_active=False) is False
