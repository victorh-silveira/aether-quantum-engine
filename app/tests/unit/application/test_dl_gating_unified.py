from src.application.services.deep_learning.dl_gating import gating_block_reason, should_execute


def test_should_execute_with_confidence_threshold():
    assert should_execute(0.80, 0.55, min_val_accuracy=0.53) is True


def test_gating_blocks_untrained_val_accuracy():
    assert gating_block_reason(0.80, 0.50, min_val_accuracy=0.53) == "val_acc"
