import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.execution_quality_gate_meta import evaluate_meta_payoff_quality
from src.application.services.execution_quality_gate_starvation import apply_starvation_edge_decay
from src.application.services.orchestrator.orchestrator_run_loop import run_orchestrator_main_loop
from src.application.services.orchestrator.trading_cycle_entry import run_trading_cycle_if_ready
from src.application.services.orchestrator.trading_cycle_entry_guards import trading_cycle_entry_allowed


def test_trading_cycle_entry_allowed_respects_cooldown():
    orch = SimpleNamespace(
        config={},
        state=SimpleNamespace(active_contracts=[]),
        _cooldown_until=time.time() + 10.0,
        running=True,
        is_trading=False,
    )
    # Com cooldown ativo no futuro, não deve permitir ciclo
    assert trading_cycle_entry_allowed(orch) is False

    # Com cooldown expirado, deve avaliar as próximas regras
    orch._cooldown_until = time.time() - 1.0
    # Mocks para passar as próximas condições
    orch._reconciliation_pending = False
    orch.last_data_signature = "sig"
    orch.get_data_state_signature = lambda: "sig_changed"
    orch._last_epoch = 100
    orch._last_processed_epoch = 50
    orch._dl_fast_cycle = True
    assert trading_cycle_entry_allowed(orch) is True


@patch(
    "src.application.services.orchestrator.trading_cycle_entry.process_redis_settlement_queue", new_callable=AsyncMock
)
@patch("src.application.services.orchestrator.trading_cycle_entry.trading_cycle_entry_allowed")
@patch("src.application.services.orchestrator.trading_cycle_entry.acquire_trading_cycle_lock", new_callable=AsyncMock)
@patch(
    "src.application.services.orchestrator.trading_cycle_entry._execute_inference_cluster_cycle", new_callable=AsyncMock
)
@patch("src.application.services.orchestrator.trading_cycle_entry.commit_trading_cycle_data_signature")
@patch("src.application.services.orchestrator.trading_cycle_entry.trading_cycle_warm_up_suspended")
@patch("src.application.services.orchestrator.trading_cycle_entry.resolve_decision_mode")
def test_run_trading_cycle_sets_cooldown_on_empty_cycle(
    mock_resolve, mock_warmup, mock_commit, mock_execute, mock_lock, mock_allowed, mock_process
):
    mock_resolve.return_value = "deep_learning"
    mock_allowed.return_value = True
    mock_lock.return_value = True
    mock_warmup.return_value = None
    # _execute_inference_cluster_cycle retorna False -> EXEC_EMPTY
    mock_execute.return_value = False

    orch = SimpleNamespace(
        config={"orchestrator": {"signature_boundary_seconds": 60}},
        state=SimpleNamespace(active_contracts=[]),
        ws=SimpleNamespace(is_running=True),
        stream=SimpleNamespace(is_synchronized=True),
        _cycle_seq=0,
        logger=MagicMock(),
        is_trading=True,
    )

    async def run():
        await run_trading_cycle_if_ready(orch)

    asyncio.run(run())

    # Deve marcar tentativa como completa
    assert hasattr(orch, "_cooldown_until")
    assert orch._cooldown_until > time.time()
    mock_commit.assert_called_once_with(orch)


def test_linear_edge_decay_limits():
    assert apply_starvation_edge_decay(0.04, 2) == pytest.approx(0.04)
    assert apply_starvation_edge_decay(-0.05, 2) == pytest.approx(-0.05)
    assert apply_starvation_edge_decay(0.04, 9) == pytest.approx(0.0)
    assert apply_starvation_edge_decay(0.04, 10) == pytest.approx(-0.01)
    assert apply_starvation_edge_decay(0.04, 15) == pytest.approx(-0.06)
    assert apply_starvation_edge_decay(0.04, 16) == pytest.approx(-0.07)


def test_evaluate_meta_payoff_quality_gbdt_waiver():
    metrics = {
        "calibrated_prob": 0.58,
        "predicted_payoff_edge": -0.80,
        "meta_classifier_applied": True,
        "meta_payoff_edge_zscore": 0.55,
        "call_votes": 6,
        "put_votes": 0,
    }
    approved = evaluate_meta_payoff_quality(
        metrics,
        exec_cfg={},
        min_payoff_edge=0.01,
        min_direction_margin=0.04,
    )
    assert approved is True
    assert metrics["execution_gate_state"] == "meta_payoff_gate_disabled"


@patch("src.application.services.orchestrator.orchestrator_run_loop.setup_session", new_callable=AsyncMock)
@patch("src.application.services.orchestrator.orchestrator_run_loop.start_streams", new_callable=AsyncMock)
@patch("src.application.services.orchestrator.orchestrator_run_loop.save_full_state", new_callable=AsyncMock)
@patch("src.application.services.orchestrator.orchestrator_run_loop.prepare_orchestrator_run_loop")
@patch("src.application.services.orchestrator.orchestrator_run_loop.await_stream_warm_up_gate", new_callable=AsyncMock)
@patch("src.application.services.orchestrator.orchestrator_run_loop.start_settlement_worker", new_callable=AsyncMock)
@patch("src.application.services.orchestrator.orchestrator_run_loop.start_ingestion_watchdog", new_callable=AsyncMock)
@patch("asyncio.sleep", new_callable=AsyncMock)
def test_run_orchestrator_loop_respects_cooldown_sleep(
    mock_sleep,
    mock_watchdog,
    mock_settlement,
    mock_warm_up,
    mock_prepare,
    mock_save,
    mock_start,
    mock_setup,
):
    mock_setup.return_value = True
    mock_start.return_value = True
    mock_warm_up.return_value = True

    orch = SimpleNamespace(
        config={"orchestrator": {"reconcile_interval_seconds": 60}},
        running=True,
        logger=MagicMock(),
        _cooldown_until=time.time() + 15.0,
        state=SimpleNamespace(active_contracts=[]),
        ws=SimpleNamespace(is_running=True),
        _tick_idle_cycle_watchdog=AsyncMock(),
        _tick_interval_cycle_if_due=AsyncMock(),
        _run_trading_cycle_if_ready=AsyncMock(),
    )

    def side_effect(seconds):
        orch.running = False

    mock_sleep.side_effect = side_effect

    async def run():
        await run_orchestrator_main_loop(orch)

    asyncio.run(run())

    mock_sleep.assert_called_once()
    args, _ = mock_sleep.call_args
    assert args[0] == pytest.approx(15.0, abs=1.0)
