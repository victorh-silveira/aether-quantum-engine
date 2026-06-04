from src.application.services.deep_learning.dl_gating import gating_block_reason, unified_execution_score


def test_unified_execution_score_raises_floor_when_deploy_ok():
    score = unified_execution_score(0.51, 0.65, deploy_ok=True, max_calibrated_raw_gap=0.12)
    assert score >= 0.53


def test_recovery_requires_brier_not_untrained_skip():
    assert (
        gating_block_reason(
            0.62,
            0.0,
            0.58,
            0.06,
            0.52,
            raw_prob=0.65,
            val_brier=1.0,
            max_val_brier=0.28,
            brier_untrained_floor=0.99,
            recovery_active=True,
            deploy_ok=True,
        )
        == "brier"
    )
