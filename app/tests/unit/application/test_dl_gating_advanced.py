from src.application.services.deep_learning.dl_gating import (
    gating_block_reason,
    resolve_gating_thresholds,
    should_execute,
)


def test_saturation_blocks_near_certainty_raw():
    assert (
        gating_block_reason(
            0.78,
            0.80,
            0.60,
            0.07,
            0.52,
            raw_prob=0.99,
            max_raw_saturation=0.90,
            saturation_min_trade_score=0.58,
        )
        == "saturation"
    )


def test_saturation_ignored_when_calibrated_score_is_weak():
    assert (
        gating_block_reason(
            0.55,
            0.83,
            0.56,
            0.05,
            0.50,
            raw_prob=0.92,
            max_raw_saturation=0.90,
            saturation_min_trade_score=0.58,
        )
        is None
    )


def test_calib_gap_blocks_inflated_score_from_live_logs():
    assert (
        gating_block_reason(
            0.96,
            0.55,
            0.58,
            0.08,
            0.52,
            raw_prob=0.24,
            max_calib_gap=0.18,
        )
        == "calib_gap"
    )
    assert (
        gating_block_reason(
            0.96,
            0.68,
            0.58,
            0.08,
            0.52,
            raw_prob=0.24,
            max_calib_gap=0.18,
        )
        is None
    )


def test_raw_conviction_floor_blocks_weak_model_side():
    assert (
        gating_block_reason(
            0.64,
            0.55,
            0.58,
            0.08,
            0.52,
            raw_prob=0.51,
            min_raw_conviction=0.52,
            deploy_ok=False,
        )
        == "raw_conviction"
    )


def test_raw_conviction_skipped_when_deploy_ok_and_calibrated_strong():
    assert (
        gating_block_reason(
            0.63,
            0.55,
            0.55,
            0.08,
            0.50,
            raw_prob=0.52,
            min_raw_conviction=0.54,
            deploy_ok=True,
            min_conviction_for_raw_bypass=0.55,
        )
        is None
    )


def test_brier_blocks_unreliable_symbol():
    assert (
        gating_block_reason(
            0.70,
            0.55,
            0.58,
            0.08,
            0.52,
            max_val_brier=0.26,
            val_brier=0.30,
        )
        == "brier"
    )


def test_recovery_thresholds_from_logs_scenario():
    params = {
        "min_conviction": 0.58,
        "min_edge_margin": 0.08,
        "min_val_accuracy": 0.52,
        "recovery_min_conviction": 0.55,
        "recovery_min_edge_margin": 0.05,
        "recovery_min_val_accuracy": 0.40,
    }
    normal = resolve_gating_thresholds(params, recovery_active=False)
    recovery = resolve_gating_thresholds(params, recovery_active=True)
    assert normal == (0.58, 0.08, 0.52)
    assert recovery == (0.55, 0.05, 0.40)
    assert should_execute(0.67, 0.40, *recovery) is True
    assert should_execute(0.67, 0.40, *normal) is False
    assert (
        should_execute(
            0.70,
            0.51,
            *normal,
            bypass_min_conviction=0.65,
            bypass_min_edge=0.12,
            allow_bypass=False,
        )
        is False
    )
    assert (
        should_execute(
            0.70,
            0.51,
            *normal,
            bypass_min_conviction=0.65,
            bypass_min_edge=0.12,
            allow_bypass=True,
        )
        is True
    )
