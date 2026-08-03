"""Cobertura residual (parte 2) apos remocao dos vetos."""

from __future__ import annotations

import json
from io import StringIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.application.services.execution_direction_fallback import build_mandatory_fallback_candidate
from src.application.services.orchestrator.execution_collect import collect_cluster_orders
from src.domain.models.trade import TradeDirection
from src.domain.risk.consensus_stake_penalty import cross_veto_recovery_waiver_allowed


class _FakePath:
    def __init__(self, payload):
        self._payload = payload

    def open(self, *_a, **_k):
        text = self._payload if isinstance(self._payload, str) else json.dumps(self._payload)
        return StringIO(text)


def test_collect_cluster_recovery_skip_symbol_retry():
    orch = SimpleNamespace(
        config={
            "orchestrator": {"execution": {}},
            "deep_learning": {},
            "risk_management": {"kelly": {}},
        },
        risk_manager=SimpleNamespace(
            pending_loss={"R_10": 1.0},
            consecutive_losses_linear=1,
            total_session_profit=-1.0,
        ),
        _active_cycle_id=1,
        _recovery_skip_counter=0,
    )
    exec_mgr = SimpleNamespace(
        orch=orch,
        logger=MagicMock(),
        _mandatory_trade_each_cycle=lambda: True,
        _trade_symbols=lambda: ["R_10"],
    )
    candidate = ("R_10", TradeDirection.CALL, {"trade_score": 0.6, "raw_prob": 0.6, "execute": True})
    with (
        patch(
            "src.application.services.orchestrator.execution_collect.gather_cluster_candidates",
            return_value=[],
        ),
        patch(
            "src.application.services.orchestrator.execution_collect.revive_ready_cluster_candidates",
            return_value=[],
        ),
        patch(
            "src.application.services.orchestrator.execution_collect.filter_loss_protection_candidates",
            side_effect=lambda c, **_k: c,
        ),
        patch(
            "src.application.services.orchestrator.execution_collect.filter_recovery_hurst_candidates",
            side_effect=lambda c, **_k: c,
        ),
        patch(
            "src.application.services.orchestrator.execution_collect.mandatory_fallback_if_empty",
            return_value=[],
        ),
        patch(
            "src.application.services.orchestrator.execution_collect.extract_collect_params",
            return_value=({}, frozenset({"R_10"}), 0.5, 0.5, 0.0, None, {}),
        ),
        patch(
            "src.application.services.orchestrator.execution_collect.resolve_mandatory_ultimate_candidate",
            side_effect=[(None, []), (candidate, [candidate])],
        ),
        patch(
            "src.application.services.orchestrator.execution_collect._select_cluster_best",
            return_value=None,
        ),
        patch(
            "src.application.services.orchestrator.execution_collect.apply_cointegration_redirect",
            side_effect=lambda c, *_a: c,
        ),
    ):
        orders = collect_cluster_orders(exec_mgr, {"R_10": {"metrics": {}}})
    assert len(orders) == 1


def test_cross_veto_recovery_waiver_delegates():
    rm = SimpleNamespace(consecutive_losses_linear=5, pending_loss={"R_10": 260.0})
    assert cross_veto_recovery_waiver_allowed({"raw_prob": 0.82}, direction="CALL", risk_manager=rm) is True


def test_mandatory_fallback_scored_none_uses_last_resort():
    orch = SimpleNamespace(_active_cycle_id=1, config={})
    decisions = {
        "R_10": {
            "direction": TradeDirection.CALL,
            "metrics": {"deploy_ok": True, "raw_prob": 0.7, "trade_score": 0.8, "val_accuracy": 0.6},
        }
    }
    last = ("R_10", TradeDirection.CALL, decisions["R_10"]["metrics"])
    with (
        patch(
            "src.application.services.execution_direction_fallback.pick_best_mandatory_candidate",
            return_value=None,
        ),
        patch(
            "src.application.services.execution_direction_fallback._scored_fallback_pick",
            return_value=None,
        ),
        patch(
            "src.application.services.execution_direction_fallback._last_resort_fallback_pick",
            return_value=last,
        ),
    ):
        assert (
            build_mandatory_fallback_candidate(
                ["R_10"],
                decisions,
                recovery_active=False,
                last_loss_symbol=None,
                orch=orch,
            )
            == last
        )


def test_dl_params_blocks_neutral_drift_from_half(monkeypatch):
    from src.application.services.deep_learning.dl_params_blocks import parse_calibration_config

    raw = {
        "method": "isotonic",
        "isotonic_min_samples": 8,
        "auto_select_by_brier": True,
        "entropy_ceiling": 0.92,
        "entropy_penalty_strength": 0.1,
        "entropy_floor": 0.5,
        "neutral_half_width": 0.08,
    }
    monkeypatch.setattr(
        "src.application.services.deep_learning.dl_params_blocks.merge_settings_block",
        lambda *_a, **_k: dict(raw),
    )
    block = parse_calibration_config({})
    assert block["calibration_neutral_drift"][0] == pytest.approx(0.42)


def test_dl_training_epochs_aux_weight_error(monkeypatch):
    from src.application.services.deep_learning import dl_training_epochs as mod

    monkeypatch.setattr(mod, "repo_path", lambda *a, **k: _FakePath({"deep_learning": {}}))
    with pytest.raises(ValueError, match="aux_regression_weight"):
        mod._aux_regression_weight()


def test_dl_calibration_bounds_missing_key(monkeypatch):
    from src.application.services.deep_learning import dl_calibration as mod

    monkeypatch.setattr(
        mod,
        "repo_path",
        lambda *a, **k: _FakePath({"deep_learning": {"calibration": {"temperature_min": 0.1}}}),
    )
    with pytest.raises(ValueError, match="obrigatorio"):
        mod._calib_bounds()


def test_dl_calibration_tolerance_missing_key(monkeypatch):
    from src.application.services.deep_learning import dl_calibration_tolerance as mod

    monkeypatch.setattr(
        mod,
        "repo_path",
        lambda *a, **k: _FakePath({"deep_learning": {"calibration": {}}}),
    )
    with pytest.raises(ValueError, match="obrigatorio"):
        mod._tol()


def test_risk_manager_cointegration_redirect_suppressed():
    from src.domain.risk.risk_manager import RiskManager

    rm = RiskManager(
        {
            "kelly": {},
            "params": {"payout_estimate": 0.95, "stake_min": 1.0},
            "soft_recovery": {
                "enabled": True,
                "micro_residual_bankroll_max": 200.0,
                "micro_residual_pending_max": 5.0,
                "micro_residual_pending_pct": 0.1,
                "coing_redirect_drawdown_threshold": 0.2,
            },
        }
    )
    rm.initial_bankroll = 100.0
    rm.pending_loss = {"R_10": 1.0}
    assert rm.cointegration_redirect_active() is False
