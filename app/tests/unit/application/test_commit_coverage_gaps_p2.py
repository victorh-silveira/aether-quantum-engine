"""Cobertura residual para commit (parte 2)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.domain.risk.consensus_recovery_gates import (
    acc_below_recovery_floor,
    adapted_blocks_dal,
    live_evidence_blocks_dal,
)
from src.domain.risk.consensus_stake_penalty import apply_soft_recovery_stake
from src.domain.risk.risk_stake_calc_helpers import apply_named_soft_stake_cap
from src.domain.risk.soft_recovery_config import load_soft_recovery_from_settings, require_soft_recovery
from src.domain.risk.stake_sizing import finalize_stake_with_min
from src.infrastructure.inference.loss_classifier_client import resolve_loss_classifier_config


def test_soft_recovery_and_stake_helpers_branches():
    soft = dict(load_soft_recovery_from_settings())
    soft["amort_cycles_min"] = 0
    with pytest.raises(ValueError, match="amort_cycles_min"):
        require_soft_recovery(soft)
    soft = dict(load_soft_recovery_from_settings())
    soft["amort_cycles_min"] = 5
    soft["amort_cycles_max"] = 2
    with pytest.raises(ValueError, match="amort_cycles_max"):
        require_soft_recovery(soft)

    assert finalize_stake_with_min(0.0, 1.0, 100.0, 0.4, recovery_linear=False) == 0.0
    assert finalize_stake_with_min(0.5, 1.0, 0.8, 0.6, recovery_linear=False) == 0.0
    assert finalize_stake_with_min(0.0, 1.0, 100.0, 0.49, recovery_linear=False, mandatory=False) == 0.0

    assert (
        apply_named_soft_stake_cap(
            10.0,
            100.0,
            {"loss_clf_soft": True, "loss_clf_soft_max_stake_pct": "bad"},
            flag="loss_clf_soft",
            pct_key="loss_clf_soft_max_stake_pct",
        )
        == 10.0
    )
    assert (
        apply_named_soft_stake_cap(
            10.0,
            100.0,
            {"loss_clf_soft": True, "loss_clf_soft_max_stake_pct": 0.0},
            flag="loss_clf_soft",
            pct_key="loss_clf_soft_max_stake_pct",
        )
        == 10.0
    )
    assert apply_named_soft_stake_cap(
        10.0,
        100.0,
        {"loss_clf_soft": True, "loss_clf_soft_max_stake_pct": 0.05},
        flag="loss_clf_soft",
        pct_key="loss_clf_soft_max_stake_pct",
    ) == pytest.approx(5.0)

    soft_cfg = dict(load_soft_recovery_from_settings())
    assert acc_below_recovery_floor({"val_accuracy": "bad"}, 2) is False
    assert live_evidence_blocks_dal({"live_wr": 0.1}, 1, soft_cfg) is False
    assert live_evidence_blocks_dal({"live_n": "x", "live_wr": "y"}, 5, soft_cfg) is False
    soft_cfg["adapted_force_explore"] = False
    assert adapted_blocks_dal({"scale_adapted": True}, 5, soft_cfg) is False
    apply_soft_recovery_stake(
        pending_total=0.0,
        base_unit=2.0,
        consecutive_losses=0,
        previous_stake=2.0,
        bankroll=100.0,
        metrics={"val_accuracy": "bad"},
        payout=0.72,
        soft_recovery=soft_cfg,
    )


def test_loss_classifier_config_high_bounds():
    with pytest.raises(ValueError, match="soft_kelly_mult_high deve estar"):
        resolve_loss_classifier_config({"soft_kelly_mult_high": 0.0})
    with pytest.raises(ValueError, match="soft_max_stake_pct_high"):
        resolve_loss_classifier_config({"soft_max_stake_pct_high": 0.06})


@pytest.mark.asyncio
async def test_ensure_cluster_history_mini_timeframe():
    from src.infrastructure.handlers.stream_handler import StreamHandler

    ws = AsyncMock()
    ws.is_running = True
    page = [{"open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05, "epoch": 2000 + i} for i in range(4)]
    ws.send = AsyncMock(return_value={"candles": page})
    sh = StreamHandler(
        ws,
        ["R_10"],
        {
            "granularity": 3600,
            "micro_granularity": 120,
            "mini_granularity": 300,
            "history_fetch_chunk": 10,
            "history_fetch_delay_seconds": 0,
            "history_fetch_symbol_delay_seconds": 0,
        },
    )
    await sh.ensure_cluster_history(4, timeframe="mini")
    assert len(sh.mini_candles["R_10"]) == 4


@pytest.mark.asyncio
async def test_trading_cycle_empty_cap_invalid_and_orch_cfg():
    from src.application.services.orchestrator.trading_cycle_entry import run_trading_cycle_if_ready

    module = "src.application.services.orchestrator.trading_cycle_entry"

    def _orch(config):
        orch = MagicMock()
        orch.config = config
        orch.anchor = "R_10"
        orch._last_epoch = 1
        orch._cycle_seq = 0
        orch.logger = MagicMock()
        orch.ws.is_running = True
        orch.stream.is_synchronized = True
        return orch

    async def _run(config):
        orch = _orch(config)
        with (
            patch(f"{module}.process_redis_settlement_queue", new=AsyncMock()),
            patch(f"{module}.trading_cycle_entry_allowed", return_value=True),
            patch(f"{module}.acquire_trading_cycle_lock", new=AsyncMock(return_value=True)),
            patch(f"{module}.resolve_decision_mode", return_value="deep_learning"),
            patch(f"{module}.trading_cycle_warm_up_suspended", return_value="OK"),
            patch(f"{module}._execute_inference_cluster_cycle", new=AsyncMock(return_value=False)),
            patch(f"{module}.force_trade_from_orch", return_value=False),
            patch(f"{module}.seconds_until_next_signature_boundary", return_value=20.0),
            patch(f"{module}.commit_trading_cycle_data_signature"),
            patch(f"{module}.clear_log_context"),
            patch(f"{module}.mark_cycle_attempt_complete"),
            patch(f"{module}.bind_log_context"),
        ):
            assert await run_trading_cycle_if_ready(orch) is True
            assert orch._cooldown_until > 0

    await _run({"orchestrator": "bad"})
    await _run({"orchestrator": {"exec_empty_retry_seconds": "bad"}})


def test_orchestrator_loss_tracker_property():
    from src.application.services.orchestrator import Orchestrator

    with patch.object(Orchestrator, "__init__", lambda self, *a, **k: None):
        orch = Orchestrator.__new__(Orchestrator)
        with patch(
            "src.application.services.orchestrator.get_direction_loss_tracker",
            return_value="tracker",
        ) as mock_get:
            assert orch.loss_tracker == "tracker"
            mock_get.assert_called_once()
