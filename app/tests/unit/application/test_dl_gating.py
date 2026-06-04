from src.application.services.deep_learning.dl_gating import (
    effective_min_val_accuracy,
    gating_block_reason,
    should_execute,
    strong_signal_bypasses_val_acc,
)


def test_should_execute_gating():
    assert should_execute(0.59, 0.55, 0.58, 0.08, 0.52) is True
    assert should_execute(0.54, 0.55, 0.58, 0.08, 0.52) is False
    assert should_execute(0.60, 0.50, 0.58, 0.08, 0.52) is False


def test_gating_block_reason_labels():
    assert gating_block_reason(0.54, 0.60, 0.58, 0.08, 0.52) == "conviction"
    assert gating_block_reason(0.58, 0.60, 0.58, 0.09, 0.52) == "edge"
    assert gating_block_reason(0.67, 0.40, 0.58, 0.08, 0.52) == "val_acc"
    assert gating_block_reason(0.67, 0.40, 0.55, 0.05, 0.40) is None


def test_untrained_metrics_skip_val_acc_gate():
    assert (
        gating_block_reason(
            0.59,
            0.0,
            0.46,
            0.01,
            0.42,
            raw_prob=0.66,
            val_brier=1.0,
            brier_untrained_floor=0.99,
            min_raw_conviction=0.46,
        )
        is None
    )


def test_bypass_lowers_val_acc_floor_but_keeps_floor():
    assert (
        gating_block_reason(
            0.70,
            0.43,
            0.58,
            0.08,
            0.52,
            bypass_min_conviction=0.65,
            bypass_min_edge=0.12,
            bypass_min_val_accuracy=0.40,
            raw_prob=0.70,
        )
        is None
    )
    assert (
        gating_block_reason(
            0.70,
            0.51,
            0.58,
            0.08,
            0.52,
            bypass_min_conviction=0.65,
            bypass_min_edge=0.12,
            bypass_min_val_accuracy=0.40,
            raw_prob=0.70,
        )
        is None
    )


def test_effective_min_val_accuracy():
    assert (
        effective_min_val_accuracy(
            0.52,
            0.55,
            bypass_min_conviction=0.53,
            bypass_min_edge=0.03,
            bypass_min_val_accuracy=0.40,
            allow_bypass=True,
        )
        == 0.40
    )
    assert (
        effective_min_val_accuracy(
            0.52,
            0.70,
            bypass_min_conviction=0.65,
            bypass_min_edge=0.12,
            bypass_min_val_accuracy=0.50,
            allow_bypass=False,
        )
        == 0.52
    )


def test_moderate_signal_lowers_val_acc_floor():
    assert (
        gating_block_reason(
            0.55,
            0.41,
            0.52,
            0.02,
            0.42,
            moderate_min_conviction=0.53,
            moderate_min_edge=0.03,
            moderate_min_val_accuracy=0.40,
        )
        is None
    )
    assert (
        gating_block_reason(
            0.55,
            0.39,
            0.52,
            0.02,
            0.42,
            moderate_min_conviction=0.53,
            moderate_min_edge=0.03,
            moderate_min_val_accuracy=0.40,
        )
        == "val_acc"
    )


def test_user_log_scenario_1hz100v_strong_signal():
    assert (
        gating_block_reason(
            0.69,
            0.45,
            0.52,
            0.02,
            0.42,
            bypass_min_conviction=0.58,
            bypass_min_edge=0.08,
        )
        is None
    )


def test_recovery_exec_none_scenario_allows_strong_and_moderate():
    recovery = (0.55, 0.05, 0.45)
    assert (
        gating_block_reason(
            0.64,
            0.38,
            *recovery,
            bypass_min_conviction=0.62,
            bypass_min_edge=0.12,
            bypass_min_val_accuracy=0.35,
            moderate_min_conviction=0.57,
            moderate_min_edge=0.07,
            moderate_min_val_accuracy=0.45,
            allow_bypass=True,
            raw_prob=0.64,
        )
        is None
    )
    assert (
        gating_block_reason(
            0.58,
            0.46,
            *recovery,
            moderate_min_conviction=0.57,
            moderate_min_edge=0.07,
            moderate_min_val_accuracy=0.45,
            allow_bypass=True,
            raw_prob=0.58,
        )
        is None
    )
    assert (
        gating_block_reason(
            0.54,
            0.46,
            *recovery,
            moderate_min_conviction=0.57,
            moderate_min_edge=0.07,
            moderate_min_val_accuracy=0.45,
            allow_bypass=True,
        )
        == "conviction"
    )


def test_strong_signal_bypasses_val_acc_from_user_log():
    assert strong_signal_bypasses_val_acc(0.70, 0.65, 0.12) is True
    assert (
        should_execute(
            0.70,
            0.51,
            0.58,
            0.08,
            0.52,
            bypass_min_conviction=0.65,
            bypass_min_edge=0.12,
        )
        is True
    )
    assert (
        should_execute(
            0.56,
            0.67,
            0.58,
            0.08,
            0.52,
            bypass_min_conviction=0.65,
            bypass_min_edge=0.12,
        )
        is False
    )
