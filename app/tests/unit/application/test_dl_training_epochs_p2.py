import pytest
import torch

from src.application.services.deep_learning.model import create_direction_model


def test_checkpoint_keeps_peak_acc_when_later_loss_improves():
    from src.application.services.deep_learning.dl_training_checkpoint import checkpoint_if_improved

    model = create_direction_model(arch="tcn")
    loss1, acc1, _sharp1, _sl1, state_peak, _sharp_state1, improved1 = checkpoint_if_improved(
        model,
        val_loss=0.80,
        val_acc=0.55,
        val_sharpness=0.05,
        min_sharpness=0.01,
        min_val_accuracy=0.53,
        best_val_loss=float("inf"),
        best_val_acc=-1.0,
        best_sharp_acc=-1.0,
        best_sharp_loss=float("inf"),
    )
    assert improved1 is True
    assert state_peak is not None
    peak_key = next(iter(state_peak))
    peak_tensor = state_peak[peak_key].clone()
    with torch.no_grad():
        for tensor in model.parameters():
            tensor.add_(0.5)
    loss2, acc2, _sharp2, _sl2, state_loss_only, _sharp_state2, improved2 = checkpoint_if_improved(
        model,
        val_loss=0.50,
        val_acc=0.51,
        val_sharpness=0.05,
        min_sharpness=0.01,
        min_val_accuracy=0.53,
        best_val_loss=loss1,
        best_val_acc=acc1,
        best_sharp_acc=acc1,
        best_sharp_loss=0.80,
    )
    assert improved2 is True
    assert state_loss_only is None
    assert acc2 == pytest.approx(0.55)
    assert loss2 == pytest.approx(0.50)
    assert torch.equal(state_peak[peak_key], peak_tensor)


def test_prefer_sharp_checkpoint_over_dull_peak():
    from src.application.services.deep_learning.dl_training_checkpoint import (
        checkpoint_if_improved,
        prefer_sharp_checkpoint,
    )

    model = create_direction_model(arch="tcn")
    _l, _a, sharp_acc, sharp_loss, dull_state, sharp_state, _imp = checkpoint_if_improved(
        model,
        val_loss=0.7,
        val_acc=0.56,
        val_sharpness=0.002,
        min_sharpness=0.01,
        min_val_accuracy=0.53,
        best_val_loss=float("inf"),
        best_val_acc=-1.0,
        best_sharp_acc=-1.0,
        best_sharp_loss=float("inf"),
    )
    assert dull_state is not None
    assert sharp_state is None
    _l2, _a2, sharp_acc2, sharp_loss2, _s2, sharp_state2, _imp2 = checkpoint_if_improved(
        model,
        val_loss=0.55,
        val_acc=0.54,
        val_sharpness=0.04,
        min_sharpness=0.01,
        min_val_accuracy=0.53,
        best_val_loss=_l,
        best_val_acc=_a,
        best_sharp_acc=sharp_acc,
        best_sharp_loss=sharp_loss,
    )
    assert sharp_state2 is not None
    assert prefer_sharp_checkpoint(dull_state, sharp_state2) is sharp_state2
    assert sharp_acc2 == pytest.approx(0.54)
    assert sharp_loss2 == pytest.approx(0.55)


def test_sharp_checkpoint_prefers_lower_val_loss():
    from src.application.services.deep_learning.dl_training_checkpoint import checkpoint_if_improved

    model = create_direction_model(arch="tcn")
    _l, _a, sa, sl, _bs, first, _i = checkpoint_if_improved(
        model,
        val_loss=0.60,
        val_acc=0.55,
        val_sharpness=0.04,
        min_sharpness=0.01,
        min_val_accuracy=0.53,
        best_val_loss=float("inf"),
        best_val_acc=-1.0,
        best_sharp_acc=-1.0,
        best_sharp_loss=float("inf"),
    )
    assert first is not None
    _l2, _a2, sa2, sl2, _bs2, second, _i2 = checkpoint_if_improved(
        model,
        val_loss=0.40,
        val_acc=0.54,
        val_sharpness=0.05,
        min_sharpness=0.01,
        min_val_accuracy=0.53,
        best_val_loss=_l,
        best_val_acc=_a,
        best_sharp_acc=sa,
        best_sharp_loss=sl,
    )
    assert second is not None
    assert sl2 == pytest.approx(0.40)
    assert sa2 == pytest.approx(0.54)
