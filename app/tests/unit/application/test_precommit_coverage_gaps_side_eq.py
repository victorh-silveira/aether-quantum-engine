from unittest.mock import MagicMock, patch

from src.application.services.execution_direction_resolver import resolve_execution_direction
from src.domain.models.trade import TradeDirection


def test_side_eq_margin_boost_blocks_and_hard_skip():
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
            "direction_margin": 0.01,
        },
    }
    soft = MagicMock(
        action="soft_penalty",
        reason="side_imbalance_large_n",
        kelly_mult=0.55,
        margin_boost=0.05,
        call_n=50,
        call_wins=20,
        put_n=50,
        put_wins=30,
        freq_bias=0.5,
        side_wr=0.4,
        z_vs_half=-1.0,
    )
    with (
        patch(
            "src.application.services.execution_direction_resolver.evaluate_proposed_side_equilibrium",
            return_value=soft,
        ),
        patch(
            "src.application.services.execution_direction_resolver.apply_side_equilibrium_to_metrics",
            side_effect=lambda metrics, decision, proposed: (
                metrics.update(
                    {
                        "side_eq_margin_boost": 0.05,
                        "quality_min_direction_margin": 0.05,
                        "direction_margin": 0.01,
                    }
                )
                or False
            ),
        ),
        patch("src.application.services.execution_direction_resolver.log_side_equilibrium"),
    ):
        assert resolve_execution_direction(entry, symbol="R_10") is None
    entry2 = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "calibrated_prob": 0.72,
            "raw_prob": 0.72,
            "deploy_ok": True,
            "execute": True,
            "predicted_payoff_edge": 0.25,
            "edge_zscore": 1.2,
            "edge_zscore_samples": 20,
            "direction_margin": 0.01,
        },
    }
    with (
        patch(
            "src.application.services.execution_direction_resolver.evaluate_proposed_side_equilibrium",
            return_value=soft,
        ),
        patch(
            "src.application.services.execution_direction_resolver.apply_side_equilibrium_to_metrics",
            return_value=True,
        ),
        patch("src.application.services.execution_direction_resolver.log_side_equilibrium"),
    ):
        assert resolve_execution_direction(entry2, symbol="R_10") is None
