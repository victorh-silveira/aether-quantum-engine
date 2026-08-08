"""Cobertura residual (parte 2) apos remocao dos vetos."""

from __future__ import annotations

import json
from io import StringIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator.orchestrator_run_loop import (
    _recovery_pending_total,
    align_exec_empty_recovery_signature_cooldown,
)
from src.application.services.orchestrator.post_settlement_cycle import _await_exec_empty_signature_alignment
from src.application.services.orchestrator.trading_cycle_entry_guards import (
    cycle_cadence_seconds,
)
from src.application.services.regime_micro_freeze import apply_regime_freeze_if_congested


class _FakePath:
    def __init__(self, payload):
        self._payload = payload

    def open(self, *_a, **_k):
        text = self._payload if isinstance(self._payload, str) else json.dumps(self._payload)
        return StringIO(text)


def test_orchestrator_run_loop_pending_map_and_bad_cap():
    orch = SimpleNamespace(risk_manager=SimpleNamespace(pending_loss=[1, 2]))
    assert _recovery_pending_total(orch) == 0.0
    orch2 = SimpleNamespace(
        _last_cycle_cluster_executed=False,
        config={"orchestrator": "bad"},
        risk_manager=SimpleNamespace(pending_loss_total=lambda: 3.0),
    )
    with patch(
        "src.application.services.orchestrator.orchestrator_run_loop.seconds_until_next_signature_boundary",
        return_value=5.0,
    ):
        assert align_exec_empty_recovery_signature_cooldown(orch2) >= 1.0


@pytest.mark.asyncio
async def test_post_settlement_exec_empty_cooldown_and_cap():
    orch = SimpleNamespace(
        risk_manager=SimpleNamespace(pending_loss_total=lambda: 8.0),
        config={"orchestrator": {"exec_empty_retry_seconds": "bad"}},
        _cooldown_until=1000.0,
    )
    with (
        patch("src.application.services.orchestrator.post_settlement_cycle.time.time", return_value=0.0),
        patch(
            "src.application.services.orchestrator.post_settlement_cycle._await_post_settlement_breath",
            new_callable=AsyncMock,
        ) as breath,
    ):
        await _await_exec_empty_signature_alignment(orch, 0.01)
        breath.assert_awaited()
    orch2 = SimpleNamespace(
        risk_manager=SimpleNamespace(pending_loss_total=lambda: 8.0),
        config={"orchestrator": "bad"},
        _cooldown_until=0.0,
    )
    with (
        patch(
            "src.application.services.orchestrator.post_settlement_cycle.seconds_until_next_signature_boundary",
            return_value=20.0,
        ),
        patch(
            "src.application.services.orchestrator.post_settlement_cycle._await_post_settlement_breath",
            new_callable=AsyncMock,
        ) as breath2,
    ):
        await _await_exec_empty_signature_alignment(orch2, 0.01)
        breath2.assert_awaited()
    orch3 = SimpleNamespace(
        risk_manager=SimpleNamespace(pending_loss_total=lambda: 8.0),
        config={"orchestrator": {"exec_empty_retry_seconds": object()}},
        _cooldown_until=0.0,
    )
    with (
        patch(
            "src.application.services.orchestrator.post_settlement_cycle.seconds_until_next_signature_boundary",
            return_value=20.0,
        ),
        patch(
            "src.application.services.orchestrator.post_settlement_cycle._await_post_settlement_breath",
            new_callable=AsyncMock,
        ),
    ):
        await _await_exec_empty_signature_alignment(orch3, 0.01)


def test_trading_cycle_cadence_bad_retry_seconds():
    orch = SimpleNamespace(
        config={"orchestrator": {"cycle_interval_seconds": 120, "exec_empty_retry_seconds": object()}},
        _last_cycle_was_exec_empty=True,
    )
    assert cycle_cadence_seconds(orch) == 45


def test_regime_micro_freeze_active_path():
    metrics = {
        "micro_indicators": {"tick_acceleration": 0.0, "bb_width": 0.01},
        "predicted_payoff_edge_zscore": -3.0,
    }
    with patch(
        "src.application.services.regime_micro_freeze.chop_congestion_regime_active",
        return_value=True,
    ):
        assert apply_regime_freeze_if_congested(metrics, persistence_filter_active=False) is True


def test_side_equilibrium_redis_async_and_timescale_task():
    client = MagicMock()
    client.hset = MagicMock(return_value=object())
    writer = MagicMock()
    writer.enqueue_trade_outcome = MagicMock(return_value=object())
    orch = SimpleNamespace(
        config={"orchestrator": {"execution": {}}},
        state_store=SimpleNamespace(client=client),
        timescale_writer=writer,
        create_task=MagicMock(),
    )
    from src.application.services.side_equilibrium_store import record_side_equilibrium_outcome

    record_side_equilibrium_outcome(orch, "R_10", direction="PUT", won=False)
    writer.enqueue_trade_outcome.side_effect = RuntimeError("ts down")
    record_side_equilibrium_outcome(orch, "R_10", direction="PUT", won=False)


def test_settlement_outcome_planned_stake():
    from src.application.services.orchestrator.settlement_outcome import process_contract_outcome

    contract = SimpleNamespace(stake=2.5, contract_id=77, direction=None)
    orch = SimpleNamespace(
        state=SimpleNamespace(balance=100.0),
        risk_manager=SimpleNamespace(
            contract_requested_stakes={},
            contract_stakes={},
            active_contract_ids=[],
            register_result=MagicMock(),
            pending_loss={},
            contract_to_symbol={77: "R_10"},
            total_session_profit=0.0,
        ),
        _contract_cycle={77: 3},
        _cluster_results=[],
        _session_wins=0,
        _session_losses=0,
        tick_count=1,
        _last_result_cycle_id=0,
        _last_settlement_outcome=None,
        _last_loss_symbol=None,
        _last_loss_direction="",
    )
    with (
        patch(
            "src.application.services.orchestrator.settlement_outcome.resolve_executed_buy_stake",
            return_value=2.5,
        ),
        patch(
            "src.application.services.orchestrator.settlement_outcome.reconcile_settlement_profit",
            return_value=1.0,
        ),
        patch(
            "src.application.services.orchestrator.settlement_outcome.bind_executed_stake_for_contract",
        ),
        patch(
            "src.application.services.orchestrator.settlement_outcome.record_symbol_outcome",
        ),
        patch(
            "src.application.services.orchestrator.settlement_outcome.record_live_signal_outcome",
        ),
        patch(
            "src.application.services.orchestrator.settlement_outcome.record_direction_outcome",
        ),
        patch(
            "src.application.services.orchestrator.settlement_outcome.record_side_equilibrium_outcome",
        ),
        patch(
            "src.application.services.orchestrator.settlement_outcome.mark_force_retrain",
        ),
    ):
        process_contract_outcome(
            orch,
            {"buy_price": 2.5, "underlying": "R_10"},
            contract,
            77,
            1.0,
            audit_raw_prob=0.6,
            audit_direction="CALL",
            log_cluster_summary=MagicMock(),
        )
    assert orch.risk_manager.contract_requested_stakes[77] == pytest.approx(2.5)


def test_meta_payoff_veto_emergency_waiver_pending_map():
    from src.domain.risk.risk_recovery_state import meta_payoff_veto_emergency_waiver

    rm = type("RM", (), {"consecutive_losses_linear": 5, "pending_loss": {"R_10": 260.0}})()
    assert meta_payoff_veto_emergency_waiver({"raw_prob": 0.18}, direction="PUT", risk_manager=rm) is True
    rm2 = SimpleNamespace(consecutive_losses_linear=5, pending_loss_total=lambda: 260.0)
    assert meta_payoff_veto_emergency_waiver({"raw_prob": 0.82}, direction="CALL", risk_manager=rm2) is True
    rm3 = SimpleNamespace(consecutive_losses_linear=0, pending_loss_total=lambda: 0.0)
    assert meta_payoff_veto_emergency_waiver({"raw_prob": 0.82}, direction="CALL", risk_manager=rm3) is False
