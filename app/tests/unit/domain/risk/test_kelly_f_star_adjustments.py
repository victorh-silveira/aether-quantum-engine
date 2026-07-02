from unittest.mock import MagicMock

from src.domain.risk.kelly_f_star_adjustments import (
    apply_consensus_entropy_f_star,
    apply_kelly_fraction_scale,
    kelly_base_with_consensus_floor,
)


def test_apply_kelly_fraction_scale_no_metrics():
    assert apply_kelly_fraction_scale(1.5, None) == 1.5


def test_kelly_base_with_consensus_floor_uses_stake_min(kelly_config):
    cfg = {
        **kelly_config["kelly"],
        "consensus_penalty_enabled": True,
        "consensus_max_cut": 0.50,
    }
    metrics = {"consensus_entropy_retention": 0.50}
    base = kelly_base_with_consensus_floor(1000.0, 0.5, metrics, cfg, 0.8, 1.0)
    assert base == 1.0


def test_apply_consensus_entropy_f_star_logs(kelly_config):
    rm = MagicMock()
    rm.kelly_config = {**kelly_config["kelly"], "consensus_penalty_enabled": True}
    rm.consecutive_losses_linear = 0
    rm.pending_loss = {}
    rm.logger = MagicMock()
    metrics = {
        "call_votes": 1,
        "put_votes": 5,
        "indicators": {"di_diff": 0.05, "cmo": -0.40, "rsi": 0.35},
    }
    out = apply_consensus_entropy_f_star(
        rm,
        0.10,
        metrics,
        "CALL",
        silent=False,
    )
    assert out < 0.10
    rm.logger.debug.assert_called()
