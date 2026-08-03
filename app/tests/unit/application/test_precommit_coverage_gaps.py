from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.application.services.execution_direction_fallback import _orch_cycle
from src.application.services.execution_direction_meta_edge import (
    _negative_edge_skip,
    _stamp_direction_resolved_cycle,
)
from src.application.services.execution_direction_resolver import _finalize_execution_metrics
from src.application.services.execution_price_zone_gate import align_or_keep_meta_side
from src.application.services.orchestrator.orchestrator_run_loop import (
    align_exec_empty_recovery_signature_cooldown,
)
from src.application.services.orchestrator.post_settlement_cycle import (
    _await_exec_empty_signature_alignment,
)
from src.application.services.orchestrator.trading_cycle_entry_guards import _cycle_cadence_seconds
from src.domain.models.trade import TradeDirection


def test_orch_cycle_none_and_value():
    assert _orch_cycle(None) == 0
    assert _orch_cycle(SimpleNamespace(_active_cycle_id=7)) == 7


def test_stamp_direction_resolved_cycle_paths():
    entry: dict = {}
    _stamp_direction_resolved_cycle(entry, 3)
    assert entry["metrics"]["_direction_resolved_cycle"] == 3
    entry2 = {"metrics": "bad"}
    _stamp_direction_resolved_cycle(entry2, 5)
    assert entry2["metrics"] == {"_direction_resolved_cycle": 5}


def test_align_or_keep_meta_side_keep_branches():
    metrics = {"price_zone_direction": "CALL", "predicted_payoff_edge": 0.12, "meta_classifier_applied": True}
    assert (
        align_or_keep_meta_side(
            TradeDirection.PUT,
            metrics,
            dl_dir=TradeDirection.PUT,
            predicted_edge=0.12,
            meta_applied=True,
        )
        == TradeDirection.PUT
    )
    metrics2 = {"predicted_payoff_edge": 0.2, "meta_classifier_applied": True}
    with patch(
        "src.application.services.execution_price_zone_meta.align_direction_to_price_zone",
        return_value=TradeDirection.CALL,
    ):
        assert (
            align_or_keep_meta_side(
                TradeDirection.PUT,
                metrics2,
                dl_dir=TradeDirection.PUT,
                predicted_edge=0.2,
                meta_applied=True,
            )
            == TradeDirection.PUT
        )


def test_align_or_keep_meta_side_soft_drift_keeps_strong_tcn_lock():
    metrics = {"calib_drift_soft": True, "direction_margin": 0.30, "calibrated_prob": 0.18}
    with patch(
        "src.application.services.execution_price_zone_meta.align_direction_to_price_zone",
        return_value=TradeDirection.CALL,
    ):
        assert (
            align_or_keep_meta_side(
                TradeDirection.PUT,
                metrics,
                dl_dir=TradeDirection.PUT,
                predicted_edge=0.2,
                meta_applied=True,
            )
            == TradeDirection.PUT
        )
    assert metrics.get("tcn_direction_lock") is True


def test_align_or_keep_meta_side_weak_margin_soft_drift_may_realign():
    metrics = {"calib_drift_soft": True, "direction_margin": 0.02, "calibrated_prob": 0.51}
    with (
        patch(
            "src.application.services.execution_price_zone_meta.align_direction_to_price_zone",
            return_value=TradeDirection.CALL,
        ),
        patch(
            "src.application.services.execution_price_zone_meta.align_direction_to_rsi_trend",
            return_value=TradeDirection.CALL,
        ) as rsi,
    ):
        assert (
            align_or_keep_meta_side(
                TradeDirection.PUT,
                metrics,
                dl_dir=TradeDirection.PUT,
                predicted_edge=-0.1,
                meta_applied=True,
            )
            == TradeDirection.CALL
        )
        rsi.assert_called_once()


def test_finalize_toxic_escape_keeps_edge():
    entry = {"metrics": {}}
    metrics = {
        "calibrated_prob": 0.7,
        "predicted_payoff_edge": 0.15,
        "trade_score": 0.7,
        "side_eq_toxic_escape": True,
    }
    with (
        patch(
            "src.application.services.execution_direction_resolver.attach_live_signal_metrics",
            return_value=None,
        ),
        patch(
            "src.application.services.execution_direction_resolver.apply_live_calib_drift_soft",
            return_value=None,
        ),
        patch(
            "src.application.services.execution_direction_resolver.apply_meta_regression_edge",
            return_value=(TradeDirection.CALL, 0.7),
        ),
        patch(
            "src.application.services.execution_direction_resolver.resolve_direction_with_side_equilibrium",
            return_value=TradeDirection.PUT,
        ),
        patch(
            "src.application.services.execution_direction_resolver.should_veto_meta_payoff_negative_zscore",
            return_value=False,
        ),
        patch(
            "src.application.services.execution_direction_resolver.is_execution_signal_vetoed",
            return_value=False,
        ),
        patch(
            "src.application.services.execution_direction_resolver.apply_price_zone_gate_with_starvation",
            return_value=None,
        ),
        patch(
            "src.application.services.execution_direction_resolver.ensure_direction_margin",
            return_value=None,
        ),
        patch(
            "src.application.services.execution_direction_resolver._negative_edge_skip",
            return_value=False,
        ),
    ):
        out = _finalize_execution_metrics(
            entry,
            metrics,
            TradeDirection.CALL,
            0.7,
            0.15,
            meta_applied=True,
            score=0.7,
            symbol="R_10",
            force=False,
            exec_cfg={"price_zone": {"enabled": False}},
        )
    assert out is not None
    assert metrics.get("side_eq_escape_edge_kept") is True


def test_negative_edge_skip_blocks_when_edge_below_floor():
    metrics = {"predicted_payoff_edge": -0.9, "meta_classifier_applied": True}
    with patch(
        "src.application.services.execution_direction_meta_edge._resolve_meta_edge_floor",
        return_value=-0.1,
    ):
        assert (
            _negative_edge_skip(
                metrics,
                -0.9,
                force=False,
                meta_applied=True,
                exec_cfg={},
            )
            is True
        )


def test_negative_edge_skip_allows_edge_at_or_above_floor():
    metrics = {"predicted_payoff_edge": -0.05, "meta_classifier_applied": True}
    with patch(
        "src.application.services.execution_direction_meta_edge._resolve_meta_edge_floor",
        return_value=-0.10,
    ):
        result = _negative_edge_skip(
            metrics,
            -0.05,
            force=False,
            meta_applied=True,
            exec_cfg={},
        )
        assert result is False
        assert metrics.get("gate_reason") is None


def test_negative_edge_skip_blocks_below_floor():
    metrics = {"predicted_payoff_edge": -0.05, "meta_classifier_applied": True}
    with patch(
        "src.application.services.execution_direction_meta_edge._resolve_meta_edge_floor",
        return_value=0.0,
    ):
        result = _negative_edge_skip(
            metrics,
            -0.05,
            force=False,
            meta_applied=True,
            exec_cfg={},
        )
        assert result is True
        assert metrics.get("gate_reason") == "meta_negative_edge"
        assert metrics.get("meta_edge_floor") == 0.0


def test_cycle_and_cooldown_invalid_cfg():
    orch = SimpleNamespace(
        config={"orchestrator": {"cycle_interval_seconds": 120, "exec_empty_retry_seconds": "bad"}},
        _last_cycle_was_exec_empty=True,
    )
    assert _cycle_cadence_seconds(orch) == 45
    orch2 = SimpleNamespace(
        config={"orchestrator": "bad"},
        _last_cycle_cluster_executed=False,
        risk_manager=SimpleNamespace(pending_loss={"R_10": 1.0}),
    )
    with patch(
        "src.application.services.orchestrator.orchestrator_run_loop.seconds_until_next_signature_boundary",
        return_value=90.0,
    ):
        assert align_exec_empty_recovery_signature_cooldown(orch2) == 45.0
    orch3 = SimpleNamespace(
        config={"orchestrator": {"exec_empty_retry_seconds": "bad"}},
        _last_cycle_cluster_executed=False,
        risk_manager=SimpleNamespace(pending_loss={"R_10": 1.0}),
    )
    with patch(
        "src.application.services.orchestrator.orchestrator_run_loop.seconds_until_next_signature_boundary",
        return_value=90.0,
    ):
        assert align_exec_empty_recovery_signature_cooldown(orch3) == 45.0


@pytest.mark.asyncio
async def test_post_settlement_invalid_cfg_paths():
    for cfg in ({"orchestrator": "bad"}, {"orchestrator": {"exec_empty_retry_seconds": "bad"}}):
        orch = SimpleNamespace(
            config=cfg,
            _cooldown_until=0.0,
            risk_manager=SimpleNamespace(pending_loss={"R_10": 1.0}),
            logger=MagicMock(),
        )
        with (
            patch(
                "src.application.services.orchestrator.post_settlement_cycle.seconds_until_next_signature_boundary",
                return_value=80.0,
            ),
            patch(
                "src.application.services.orchestrator.post_settlement_cycle._await_post_settlement_breath",
                return_value=None,
            ),
        ):
            await _await_exec_empty_signature_alignment(orch, poll=0.01)
        assert orch._cooldown_until > 0.0
