"""Testes dos gates de EXPLORE no soft recovery."""

from src.domain.risk.consensus_recovery_gates import (
    acc_below_recovery_floor,
    adapted_blocks_dal,
    conviction_below_recovery_ladder,
    live_evidence_blocks_dal,
    metric_hurst,
    resolve_recovery_force_explore,
)
from src.domain.risk.soft_recovery_config import load_soft_recovery_from_settings


def test_metric_hurst_reads_indicators_bucket():
    assert metric_hurst({"indicators": {"hurst": 0.41}}) == 0.41
    assert metric_hurst({"regime_chop_hurst": 0.39}) == 0.39
    assert metric_hurst({"hurst": 0.5}) == 0.5
    assert metric_hurst({}) is None
    assert metric_hurst(None) is None
    assert metric_hurst({"hurst": "bad", "indicators": {"hurst": "nope"}}) is None
    assert metric_hurst({"hurst": "bad", "indicators": {"hurst": 0.44}}) == 0.44


def test_recovery_gate_helpers_branches():
    soft = load_soft_recovery_from_settings()
    assert acc_below_recovery_floor({"val_accuracy": 0.01}, 3) is True
    assert live_evidence_blocks_dal({"live_n": 20, "live_wr": 0.40}, 3, soft) is True
    assert adapted_blocks_dal({"scale_adapted": True}, int(soft["adapted_force_explore_linear_min"]), soft) is True


def test_conviction_below_recovery_ladder_detects_weak_signal():
    assert conviction_below_recovery_ladder(None, 1) is False
    assert conviction_below_recovery_ladder({}, 1) is False
    assert conviction_below_recovery_ladder({"calibrated_prob": 0.5884}, 1) is True
    assert conviction_below_recovery_ladder({"calibrated_prob": 0.65}, 1) is False
    assert conviction_below_recovery_ladder({"p_exec": 0.4116}, 1) is True
    assert conviction_below_recovery_ladder({"prob": 0.30}, 1) is False


def test_resolve_recovery_force_explore_returns_expected_flags():
    soft = load_soft_recovery_from_settings()
    acc, live, adapted, quality, force_early = resolve_recovery_force_explore(
        {"val_accuracy": 0.01}, 3, soft, material_pending=True, cover_enabled=True
    )
    assert acc is True
    assert quality is False
    assert force_early is False

    acc, live, adapted, quality, force_early = resolve_recovery_force_explore(
        {"val_accuracy": 0.01}, 3, soft, material_pending=False, cover_enabled=True
    )
    assert quality is True
    assert force_early is True


def test_should_downgrade_recovery_stake():
    from src.domain.risk.consensus_recovery_gates import should_downgrade_recovery_stake

    assert should_downgrade_recovery_stake(None, 1) is False
    assert should_downgrade_recovery_stake({}, 1) is False
    assert should_downgrade_recovery_stake({"calibrated_prob": 0.5884}, 1) is True
    assert (
        should_downgrade_recovery_stake(
            {"calibrated_prob": 0.65, "closed_micro_candle_dir": "CALL", "exec_direction": "CALL"}, 1
        )
        is False
    )
    assert (
        should_downgrade_recovery_stake(
            {"calibrated_prob": 0.65, "closed_micro_candle_dir": "PUT", "exec_direction": "CALL"}, 1
        )
        is True
    )
    assert (
        should_downgrade_recovery_stake(
            {"calibrated_prob": 0.65, "closed_micro_candle_dir": "PUT", "exec_direction": "CALL"}, 0
        )
        is False
    )
