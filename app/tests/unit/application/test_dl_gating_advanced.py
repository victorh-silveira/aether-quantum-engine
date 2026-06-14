from src.application.services.deep_learning.dl_gating import gating_block_reason, should_execute


def test_gating_allows_at_threshold_boundary():
    assert should_execute(0.75, 0.53, min_val_accuracy=0.53, call_threshold=0.75, put_threshold=0.25)


def test_gating_blocks_just_below_call_threshold():
    assert gating_block_reason(0.749, 0.60, call_threshold=0.75, put_threshold=0.25) == "confidence"
