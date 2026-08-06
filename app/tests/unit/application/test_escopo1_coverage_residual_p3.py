"""Cobertura residual (parte 2) apos remocao dos vetos."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from src.application.services.deep_learning.dl_calibration import CalibratorState, apply_calibrator_stable
from src.application.services.deep_learning.dl_calibration_fit import _select_best_calibrator
from src.application.services.deep_learning.dl_gate_config import describe_deploy_block
from src.application.services.deep_learning.dl_sharpness import (
    assert_export_sharpness_floor,
    assert_export_sharpness_value,
    mean_sharpness,
)
from src.application.services.deep_learning.dl_symbol_runtime import _persist_deploy_ok_flag
from src.application.services.orchestrator.orchestrator_run_loop import (
    _recovery_pending_total,
    align_exec_empty_recovery_signature_cooldown,
    run_orchestrator_main_loop,
)
from src.application.services.orchestrator.post_settlement_cycle import _await_exec_empty_signature_alignment
from src.application.services.orchestrator.regime_freeze_yield import propagate_cluster_signal_suspended
from src.application.services.regime_micro_freeze import apply_regime_freeze_if_congested
from src.application.services.side_equilibrium_store import snapshot_side_counts
from src.domain.analytics.side_equilibrium import SideCounts


class _FakePath:
    def __init__(self, payload):
        self._payload = payload

    def open(self, *_a, **_k):
        text = self._payload if isinstance(self._payload, str) else json.dumps(self._payload)
        return StringIO(text)


def test_recovery_pending_and_align_exec_empty():
    orch = SimpleNamespace(risk_manager=None)
    assert _recovery_pending_total(orch) == 0.0
    orch2 = SimpleNamespace(risk_manager=SimpleNamespace(pending_loss={"a": 1.5, "b": 2.0}))
    assert _recovery_pending_total(orch2) == pytest.approx(3.5)
    orch3 = SimpleNamespace(
        _last_cycle_cluster_executed=False,
        config={"orchestrator": {"exec_empty_retry_seconds": "bad"}},
        risk_manager=SimpleNamespace(pending_loss_total=lambda: 10.0),
    )
    with patch(
        "src.application.services.orchestrator.orchestrator_run_loop.seconds_until_next_signature_boundary",
        return_value=100.0,
    ):
        delay = align_exec_empty_recovery_signature_cooldown(orch3)
    assert delay == pytest.approx(45.0)


@pytest.mark.asyncio
async def test_await_exec_empty_signature_alignment_paths():
    orch = SimpleNamespace(
        risk_manager=SimpleNamespace(pending_loss={"OTC_SPC": 5.0}),
        config={"orchestrator": {"exec_empty_retry_seconds": 45}},
        _cooldown_until=0.0,
    )
    with (
        patch(
            "src.application.services.orchestrator.post_settlement_cycle.seconds_until_next_signature_boundary",
            return_value=30.0,
        ),
        patch(
            "src.application.services.orchestrator.post_settlement_cycle._await_post_settlement_breath",
            new_callable=AsyncMock,
        ) as breath,
    ):
        await _await_exec_empty_signature_alignment(orch, 0.01)
        breath.assert_awaited()
    orch2 = SimpleNamespace(risk_manager=None)
    with patch(
        "src.application.services.orchestrator.post_settlement_cycle._poll_delay",
        new_callable=AsyncMock,
    ) as poll:
        await _await_exec_empty_signature_alignment(orch2, 0.01)
        poll.assert_awaited()


@pytest.mark.asyncio
async def test_run_orchestrator_main_loop_cooldown_branch():
    orch = MagicMock()
    orch.running = True
    orch.config = {"orchestrator": {}}
    orch.ws.is_running = True
    orch.last_data_signature = "sig"
    orch._cooldown_until = 0.0
    orch._tick_idle_cycle_watchdog = AsyncMock()
    orch._tick_interval_cycle_if_due = AsyncMock()
    orch._run_trading_cycle_if_ready = AsyncMock()
    with (
        patch(
            "src.application.services.orchestrator.orchestrator_run_loop.setup_session",
            AsyncMock(return_value=True),
        ),
        patch(
            "src.application.services.orchestrator.orchestrator_run_loop.start_streams",
            AsyncMock(return_value=True),
        ),
        patch(
            "src.application.services.orchestrator.orchestrator_run_loop.prepare_orchestrator_run_loop",
        ),
        patch(
            "src.application.services.orchestrator.orchestrator_run_loop.await_stream_warm_up_gate",
            AsyncMock(),
        ),
        patch(
            "src.application.services.orchestrator.orchestrator_run_loop.start_settlement_worker",
            AsyncMock(),
        ),
        patch(
            "src.application.services.orchestrator.orchestrator_run_loop.start_ingestion_watchdog",
            AsyncMock(),
        ),
        patch(
            "src.application.services.orchestrator.orchestrator_run_loop.align_exec_empty_recovery_signature_cooldown",
            return_value=0.0,
        ),
        patch(
            "src.application.services.orchestrator.orchestrator_run_loop.get_data_state_signature",
            return_value="sig",
        ),
        patch(
            "src.application.services.orchestrator.orchestrator_run_loop._enforce_post_settlement_deadlock_exit",
        ),
        patch("src.application.services.orchestrator.orchestrator_run_loop.time.time", side_effect=[0.0, 0.0, 100.0]),
        patch("asyncio.sleep", new_callable=AsyncMock) as sleep_mock,
    ):
        orch._cooldown_until = 50.0
        orch.running = True

        async def stop_after_sleep(*_a, **_k):
            orch.running = False

        sleep_mock.side_effect = stop_after_sleep
        await run_orchestrator_main_loop(orch)
    sleep_mock.assert_awaited()


def test_regime_freeze_propagate_and_micro_freeze():
    decisions = {"x": "bad", "OTC_SPC": {"metrics": "bad"}}
    propagate_cluster_signal_suspended(decisions)
    assert decisions["OTC_SPC"]["metrics"]["signal_status"] == "SIGNAL_SUSPENDED"
    metrics = {
        "micro_indicators": {"tick_acceleration": 0.0, "bb_width": 0.01},
        "predicted_payoff_edge_zscore": -2.0,
    }
    with patch(
        "src.application.services.regime_micro_freeze.chop_congestion_regime_active",
        return_value=True,
    ):
        assert apply_regime_freeze_if_congested(metrics, persistence_filter_active=False) is True


def test_side_equilibrium_store_branches():
    orch = SimpleNamespace(_side_equilibrium_hist=None, config={"orchestrator": {"execution": {}}})
    assert snapshot_side_counts(orch, "OTC_SPC", window=5).total == 0
    assert SideCounts(call_n=0).wr("CALL") is None
    client = MagicMock()
    client.hset = MagicMock(return_value=object())
    orch2 = SimpleNamespace(
        config={"orchestrator": {"execution": {}}},
        state_store=SimpleNamespace(client=client),
        timescale_writer=SimpleNamespace(enqueue_trade_outcome=MagicMock(return_value=object())),
        create_task=MagicMock(),
    )
    from src.application.services.side_equilibrium_store import record_side_equilibrium_outcome

    record_side_equilibrium_outcome(orch2, "OTC_SPC", direction="CALL", won=True)
    client.hset = MagicMock(return_value=None)


def test_dl_sharpness_gate_and_persist_errors(tmp_path: Path):
    assert mean_sharpness([]) == pytest.approx(0.0)
    with pytest.raises(RuntimeError, match="Export TCN bloqueado"):
        assert_export_sharpness_value(0.01, floor=0.05)
    with pytest.raises(RuntimeError):
        assert_export_sharpness_floor([0.51], floor=0.05)
    assert describe_deploy_block(
        mini_ok=True, val_accuracy=0.56, val_brier=0.20, gate_cfg={"soft_min_val_accuracy": 0.53}
    ) == ("mini_ok mas gate rejeitou (inesperado)")
    assert describe_deploy_block(
        mini_ok=False,
        val_accuracy=0.56,
        val_brier=0.20,
        gate_cfg={"soft_min_val_accuracy": 0.53, "soft_max_brier": 0.26},
    ) == ("gate rejeitou sem motivo tipado")
    bad = tmp_path / "bad.pth"
    _persist_deploy_ok_flag(bad, deploy_ok=True)
    torch.save("not-dict", bad)
    _persist_deploy_ok_flag(bad, deploy_ok=True)
    payload = {"deploy_ok": True}
    torch.save(payload, bad)
    _persist_deploy_ok_flag(bad, deploy_ok=True)
    with patch("src.application.services.deep_learning.dl_symbol_runtime.torch.save", side_effect=RuntimeError("disk")):
        _persist_deploy_ok_flag(bad, deploy_ok=False)


def test_dl_calibration_fit_and_stable_margin():
    cal = CalibratorState(method="isotonic", isotonic_x=(0.2, 0.8), isotonic_y=(0.1, 0.9))
    with patch(
        "src.application.services.deep_learning.dl_calibration.apply_calibrator",
        return_value=0.52,
    ):
        out = apply_calibrator_stable(0.7, cal, margin_floor=0.15)
    assert out == pytest.approx(0.7)
    candidates = [(CalibratorState(method="platt"), 0.2, 0.1, 0.01)]
    picked = _select_best_calibrator(candidates, min_sharpness=0.5)
    assert picked.method == "platt"
