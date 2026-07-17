from src.application.services.execution_loss_protection import (
    _loss_protection_cfg,
    _loss_protection_recovery_blocks,
    _loss_protection_signal_blocks,
    filter_loss_protection_candidates,
)
from src.domain.models.trade import TradeDirection


def test_loss_protection_helpers_still_evaluate_blocks():
    assert _loss_protection_cfg(None) == {}
    assert _loss_protection_cfg({"loss_protection": "bad"}) == {}
    assert _loss_protection_cfg({"loss_protection": {"min_direction_margin": 0.2}})["min_direction_margin"] == 0.2
    metrics = {"indicators": {"hurst": 0.40}, "direction_margin": 0.10, "edge": 0.90, "edge_zscore": 1.0}
    assert (
        _loss_protection_recovery_blocks(
            metrics,
            recovery_active=True,
            linear=2,
            min_margin_recovery=0.20,
            recovery_min_hurst=0.50,
            margin=0.10,
        )
        is True
    )
    assert (
        _loss_protection_recovery_blocks(
            {"indicators": {"hurst": 0.40}},
            recovery_active=True,
            linear=1,
            min_margin_recovery=0.20,
            recovery_min_hurst=0.50,
            margin=0.30,
        )
        is True
    )
    assert (
        _loss_protection_recovery_blocks(
            {"micro_indicators": {"hurst": 0.40}},
            recovery_active=True,
            linear=1,
            min_margin_recovery=0.20,
            recovery_min_hurst=0.50,
            margin=0.30,
        )
        is True
    )
    assert (
        _loss_protection_recovery_blocks(
            {},
            recovery_active=True,
            linear=1,
            min_margin_recovery=0.20,
            recovery_min_hurst=0.50,
            margin=0.30,
        )
        is False
    )
    assert (
        _loss_protection_signal_blocks(
            metrics,
            TradeDirection.PUT,
            margin=0.10,
            edge=0.90,
            z_edge=1.0,
            min_margin=0.18,
            max_edge_low_margin=0.40,
            max_z_low_margin=0.85,
        )
        is True
    )
    assert (
        _loss_protection_signal_blocks(
            {"edge": 0.10, "direction_margin": 0.10, "edge_zscore": 1.0, "raw_prob": 0.5, "calibrated_prob": 0.5},
            TradeDirection.PUT,
            margin=0.10,
            edge=0.10,
            z_edge=1.0,
            min_margin=0.18,
            max_edge_low_margin=0.40,
            max_z_low_margin=0.85,
        )
        is True
    )
    assert (
        _loss_protection_signal_blocks(
            {
                "edge": 0.55,
                "direction_margin": 0.20,
                "edge_zscore": 0.2,
                "raw_prob": 0.80,
                "calibrated_prob": 0.80,
            },
            TradeDirection.PUT,
            margin=0.20,
            edge=0.55,
            z_edge=0.2,
            min_margin=0.18,
            max_edge_low_margin=0.40,
            max_z_low_margin=0.85,
        )
        is True
    )
    assert (
        _loss_protection_recovery_blocks(
            {},
            recovery_active=False,
            linear=0,
            min_margin_recovery=0.20,
            recovery_min_hurst=0.50,
            margin=0.30,
        )
        is False
    )
    empty = filter_loss_protection_candidates(
        [],
        exec_cfg={},
        recovery_active=False,
        consecutive_losses=0,
    )
    assert empty == []
    invalid_only = filter_loss_protection_candidates(
        [("RDBULL", TradeDirection.PUT)],
        exec_cfg={},
        recovery_active=False,
        consecutive_losses=0,
    )
    assert invalid_only == [("RDBULL", TradeDirection.PUT)]
