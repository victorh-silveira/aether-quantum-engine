"""Fluxo de checkpoint, recovery, proposta e liquidacao com teto unico."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from scripts.operations.check_dl_checkpoint import evaluate_checkpoint
from src.application.services.deep_learning.dl_features import FEATURE_DIM
from src.application.services.deep_learning.dl_runtime_deploy_gate import apply_deploy_gate
from src.application.services.deep_learning.model import create_direction_model
from src.application.services.orchestrator.execution_orders import place_order
from src.application.services.orchestrator.settlement_outcome import process_contract_outcome
from src.domain.models.trade import Contract, TradeDirection, TradeStatus
from src.domain.risk.risk_stake_calc_helpers import cap_provisional_stake


@pytest.mark.asyncio
async def test_checkpoint_recovery_proposal_and_settlement_obey_one_percent_cap(tmp_path):
    checkpoint = tmp_path / "1HZ75V.pth"
    torch.save(
        {
            "state_dict": create_direction_model(arch="tcn").state_dict(),
            "norm_mean": [0.0] * FEATURE_DIM,
            "norm_std": [1.0] * FEATURE_DIM,
            "feature_dim": FEATURE_DIM,
            "lookback": 32,
            "granularity": 300,
            "label_horizon_bars": 1,
            "label_mode": "spot_forward",
        },
        checkpoint,
    )
    settings = {
        "deep_learning": {"lookback": 32, "label_horizon_bars": 1, "label_mode": "spot_forward"},
        "data_handler": {"micro_granularity": 300},
    }
    checkpoint_ready, _ = evaluate_checkpoint(checkpoint, settings=settings)
    assert checkpoint_ready
    entry = apply_deploy_gate(
        {"metrics": {"execute": True}},
        {"checkpoint_loaded": checkpoint_ready, "session_trained": checkpoint_ready},
        {"checkpoint_max_stake_pct": 0.5},
    )
    metrics = entry["metrics"]
    metrics["recovery_cap_mode"] = "cover_l0"
    assert metrics["checkpoint_exploration"] is True
    assert metrics["provisional_max_stake_pct"] == 0.01
    assert cap_provisional_stake(350.0, 10000.0, metrics, safe_cap=350.0) == 100.0

    risk_manager = MagicMock()
    risk_manager.contract_to_symbol = {}
    risk_manager.contract_requested_stakes = {}
    risk_manager.contract_stakes = {}
    risk_manager.active_contract_ids = []
    orch = MagicMock()
    orch._active_cycle_id = 1
    orch.trading_transport = "rest"
    orch.deriv_account_id = "DOT1"
    orch.state = SimpleNamespace(balance=10000.0)
    orch.risk_manager = risk_manager
    orch.config = {
        "risk_management": {"params": {"stake_min": 1.0, "duration": 5, "duration_unit": "m"}},
        "orchestrator": {"execution": {}},
    }
    orch.auth.rest_client.return_value.list_accounts = AsyncMock(return_value=[])
    contract = Contract(
        contract_id=77,
        proposal_id="proposal",
        status=TradeStatus.OPEN,
        buy_price=100.0,
        payout=185.0,
        symbol="1HZ75V",
        direction=TradeDirection.CALL,
        stake=100.0,
        expiry_time=300,
        longcode="Rise/Fall",
    )
    orch.trade_handler.buy_with_parameters = AsyncMock(return_value=contract)
    with patch("src.application.services.orchestrator.execution_orders.schedule_rest_contract_settlement") as schedule:
        bought = await place_order(MagicMock(orch=orch), "1HZ75V", TradeDirection.CALL, 350.0, metrics=metrics)
    assert bought is contract
    assert orch.trade_handler.buy_with_parameters.await_args.args[2] == 100.0
    schedule.assert_called_once()

    orch.tick_count = 300
    orch._cluster_results = []
    orch._contract_cycle = {77: 1}
    orch._session_wins = 0
    orch._session_losses = 0
    summary = MagicMock()
    with (
        patch("src.application.services.orchestrator.settlement_outcome.record_symbol_outcome"),
        patch("src.application.services.orchestrator.settlement_outcome._record_directional_learning"),
    ):
        process_contract_outcome(
            orch,
            {"buy_price": 100.0, "balance_after": 9900.0, "underlying": "1HZ75V"},
            contract,
            77,
            -100.0,
            log_cluster_summary=summary,
        )
    risk_manager.register_result.assert_called_once_with(
        -100.0, 77, symbol="1HZ75V", current_tick=300, direction="CALL"
    )
    assert orch._session_losses == 1
    assert orch._last_settlement_outcome == "LOSS"
