"""Cobertura residual (parte 2) apos remocao dos vetos."""

from __future__ import annotations

import json
import logging
from io import StringIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import torch

from src.application.services.deep_learning.dl_symbol_train_success import apply_successful_symbol_train
from src.application.services.orchestrator.execution_manager_execute import execute_cluster_orders
from src.domain.models.trade import TradeDirection


class _FakePath:
    def __init__(self, payload):
        self._payload = payload

    def open(self, *_a, **_k):
        text = self._payload if isinstance(self._payload, str) else json.dumps(self._payload)
        return StringIO(text)


def test_apply_successful_symbol_train_deploy_warning(tmp_path):
    ckpt = tmp_path / "R_10.pth"
    torch.save({"deploy_ok": True, "val_brier": 0.22}, ckpt)
    runtime = {"val_accuracy": 0.56, "val_brier": 0.27}
    train_result = SimpleNamespace(
        norm_stats=MagicMock(),
        val_accuracy=0.56,
        val_brier=0.27,
        calibrator=None,
        val_ece=0.1,
        avg_loss=0.4,
        epochs_ran=40,
        oos_sharpness=0.05,
    )
    gate_cfg = {"enabled": True, "soft_min_val_accuracy": 0.53, "soft_max_brier": 0.26}
    dl_config = {"model_path_template": "x/{symbol}.pth", "deploy_gate": gate_cfg}
    orch = MagicMock()
    with (
        patch(
            "src.application.services.deep_learning.dl_symbol_train_success.evaluate_mini_deploy",
            return_value=(False, 0.5, 0.27),
        ),
        patch(
            "src.application.services.deep_learning.dl_symbol_train_success.save_model_checkpoint",
        ),
        patch(
            "src.application.services.deep_learning.dl_symbol_train_success.schedule_model_upload",
        ),
        patch(
            "src.application.services.deep_learning.dl_symbol_train_success.resolve_dl_model_path",
            return_value=ckpt,
        ),
        patch(
            "src.application.services.deep_learning.dl_symbol_train_success.resolve_deploy_ok",
            return_value=False,
        ),
    ):
        apply_successful_symbol_train(
            "R_10",
            runtime,
            train_result,
            orch=orch,
            model=MagicMock(),
            prices=np.linspace(1.0, 2.0, 80),
            norm_stats=train_result.norm_stats,
            params={"lookback": 32, "arch": "tcn"},
            dl_config=dl_config,
            gate_cfg=gate_cfg,
            candle_epoch_value=1,
            granularity=60,
            level=logging.INFO,
            started=0.0,
        )
    assert runtime.get("checkpoint_preserved") is True
    assert runtime.get("session_trained") is False
    assert runtime.get("export_ok") is False
    assert torch.load(ckpt, map_location="cpu", weights_only=False)["deploy_ok"] is False


def test_demote_preserved_checkpoint_branches(tmp_path):
    from src.application.services.deep_learning.dl_symbol_train_success import _demote_preserved_checkpoint

    missing = tmp_path / "missing.pth"
    _demote_preserved_checkpoint(missing)
    bad = tmp_path / "bad.pth"
    torch.save([1, 2, 3], bad)
    _demote_preserved_checkpoint(bad)
    already = tmp_path / "already.pth"
    torch.save({"deploy_ok": False}, already)
    _demote_preserved_checkpoint(already)
    with patch("torch.load", side_effect=RuntimeError("boom")):
        _demote_preserved_checkpoint(already)


@pytest.mark.asyncio
async def test_execute_cluster_orders_force_and_reversal_stake():
    executor = MagicMock()
    executor.orch = SimpleNamespace(
        _active_cycle_id=1,
        config={
            "deep_learning": {"max_val_brier_execute": 0.28},
            "risk_management": {"params": {"duration": 120}, "kelly": {"neutral_bankroll_pct": 0.002}},
            "orchestrator": {"execution": {"force_trade_every_cycle": True}},
        },
        risk_manager=SimpleNamespace(
            kelly_config={"neutral_bankroll_pct": 0.002, "stop_win_kelly_enabled": True},
            pending_loss={},
            consecutive_losses_linear=2,
            pending_loss_total=lambda: 5.0,
            calculate_stake=MagicMock(return_value=0.0),
            register_entry_conviction=MagicMock(),
            record_contract_stake=MagicMock(),
            active_contract_ids=[],
        ),
    )
    executor._mandatory_trade_each_cycle = MagicMock(return_value=True)
    executor._place_order = AsyncMock(
        return_value=SimpleNamespace(contract_id=901, buy_price=2.5),
    )
    executor._log_exec = MagicMock()
    executor.orch.state = SimpleNamespace(add_contract=AsyncMock())
    executor.orch._contract_cycle = {}
    with (
        patch(
            "src.application.services.orchestrator.execution_manager_execute.force_trade_from_orch",
            return_value=True,
        ),
        patch(
            "src.application.services.orchestrator.execution_manager_execute.resolve_force_min_stake",
            return_value=2.5,
        ),
    ):
        count = await execute_cluster_orders(
            executor,
            [("R_10", TradeDirection.CALL, {"execute": True})],
            0.0,
            1000.0,
        )
    assert count == 1


@pytest.mark.asyncio
async def test_execute_cluster_orders_reversal_stake_floor():
    executor = MagicMock()
    executor.orch = SimpleNamespace(
        _active_cycle_id=1,
        config={
            "deep_learning": {"max_val_brier_execute": 0.28},
            "risk_management": {"params": {"duration": 120}, "kelly": {"neutral_bankroll_pct": 0.002}},
            "orchestrator": {"execution": {}},
        },
        risk_manager=SimpleNamespace(
            kelly_config={"neutral_bankroll_pct": 0.002, "stop_win_kelly_enabled": True},
            pending_loss={"R_10": 1.0},
            consecutive_losses_linear=2,
            pending_loss_total=lambda: 5.0,
            calculate_stake=MagicMock(return_value=0.0),
            register_entry_conviction=MagicMock(),
            record_contract_stake=MagicMock(),
            active_contract_ids=[],
        ),
    )
    executor._mandatory_trade_each_cycle = MagicMock(return_value=True)
    executor._place_order = AsyncMock(
        return_value=SimpleNamespace(contract_id=902, buy_price=3.0),
    )
    executor._log_exec = MagicMock()
    executor.orch.state = SimpleNamespace(add_contract=AsyncMock())
    executor.orch._contract_cycle = {}
    metrics = {"reversal_stake_floor": True, "execute": True}
    with patch(
        "src.application.services.orchestrator.execution_manager_execute.force_trade_from_orch",
        return_value=False,
    ):
        count = await execute_cluster_orders(
            executor,
            [("R_10", TradeDirection.CALL, metrics)],
            0.0,
            1000.0,
        )
    assert count == 1
