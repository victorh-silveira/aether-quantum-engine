from unittest.mock import patch

from src.application.services.execution_direction_resolver import (
    _finalize_execution_metrics,
    resolve_execution_direction,
)
from src.domain.models.trade import TradeDirection


def test_side_eq_both_sides_blocked_returns_none():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "calibrated_prob": 0.72,
            "raw_prob": 0.72,
            "deploy_ok": True,
            "execute": True,
            "predicted_payoff_edge": 0.25,
            "edge_zscore": 1.2,
            "edge_zscore_samples": 20,
            "direction_margin": 0.22,
        },
    }
    with patch(
        "src.application.services.execution_direction_resolver.resolve_direction_with_side_equilibrium",
        return_value=None,
    ):
        assert resolve_execution_direction(entry, symbol="R_10") is None


def test_side_eq_hard_skip_flips_to_opposite_under_force_trade():
    entry = {
        "direction": TradeDirection.PUT,
        "metrics": {
            "calibrated_prob": 0.48,
            "raw_prob": 0.48,
            "deploy_ok": True,
            "execute": True,
            "predicted_payoff_edge": 0.10,
            "edge_zscore": 0.5,
            "edge_zscore_samples": 10,
            "direction_margin": 0.02,
        },
    }
    with patch(
        "src.application.services.execution_direction_resolver.resolve_direction_with_side_equilibrium",
        return_value=TradeDirection.CALL,
    ):
        resolved = resolve_execution_direction(
            entry,
            symbol="R_10",
            exec_cfg={"force_trade_every_cycle": True},
        )
    assert resolved is not None
    assert resolved[0] == TradeDirection.CALL


def test_side_eq_soft_margin_blocks_after_flip_pass():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "calibrated_prob": 0.52,
            "raw_prob": 0.52,
            "deploy_ok": True,
            "execute": True,
            "predicted_payoff_edge": 0.10,
            "edge_zscore": 0.5,
            "edge_zscore_samples": 10,
            "direction_margin": 0.01,
        },
    }

    def _soft(_orch, _symbol, proposed, metrics):
        metrics["side_eq_margin_boost"] = 0.05
        metrics["quality_min_direction_margin"] = 0.10
        return proposed

    with (
        patch(
            "src.application.services.execution_direction_resolver.resolve_direction_with_side_equilibrium",
            side_effect=_soft,
        ),
        patch(
            "src.application.services.execution_direction_resolver.ensure_direction_margin",
            side_effect=lambda metrics: metrics.__setitem__("direction_margin", 0.02) or 0.02,
        ),
    ):
        assert resolve_execution_direction(entry, symbol="R_10") is None


def test_finalize_blocks_when_side_eq_margin_boost_exceeds_margin():
    entry = {"direction": TradeDirection.CALL, "metrics": {}}
    metrics = {
        "calibrated_prob": 0.55,
        "raw_prob": 0.55,
        "side_eq_margin_boost": 0.05,
        "quality_min_direction_margin": 0.08,
        "direction_margin": 0.02,
        "meta_veto_mode": "none",
    }

    def _keep_margin(m):
        m["direction_margin"] = 0.02

    with (
        patch(
            "src.application.services.execution_direction_resolver.resolve_direction_with_side_equilibrium",
            return_value=TradeDirection.CALL,
        ),
        patch(
            "src.application.services.execution_direction_resolver.ensure_direction_margin",
            side_effect=_keep_margin,
        ),
        patch(
            "src.application.services.execution_direction_resolver.apply_meta_regression_edge",
            return_value=(TradeDirection.CALL, 0.55),
        ),
        patch(
            "src.application.services.execution_direction_resolver.should_veto_meta_payoff_negative_zscore",
            return_value=False,
        ),
        patch(
            "src.application.services.execution_direction_resolver.is_execution_signal_vetoed",
            return_value=False,
        ),
    ):
        assert (
            _finalize_execution_metrics(
                entry,
                metrics,
                TradeDirection.CALL,
                0.55,
                0.1,
                meta_applied=True,
                score=0.55,
                symbol="R_10",
            )
            is None
        )
    assert metrics["gate_reason"] == "side_imbalance_large_n_margin"
    assert metrics["quality_guard_reject"] is True
