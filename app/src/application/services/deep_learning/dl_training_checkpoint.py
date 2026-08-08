"""Selecao de checkpoint de treino TCN (val_acc + sharpness + val_loss)."""

from __future__ import annotations


def checkpoint_if_improved(
    model,
    *,
    val_loss: float,
    val_acc: float,
    val_sharpness: float,
    min_sharpness: float,
    min_val_accuracy: float,
    best_val_loss: float,
    best_val_acc: float,
    best_sharp_acc: float,
    best_sharp_loss: float,
) -> tuple[float, float, float, float, dict | None, dict | None, bool]:
    """Atualiza picos de loss/acc e o melhor estado sharp com menor val_loss."""
    loss_improved = val_loss + 1e-9 < best_val_loss
    acc_improved = val_acc > best_val_acc + 1e-6
    if loss_improved:
        best_val_loss = val_loss
    best_state = None
    best_sharp_state = None
    if acc_improved:
        best_val_acc = val_acc
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    sharp_ok = float(val_sharpness) + 1e-12 >= float(min_sharpness)
    acc_floor_ok = float(val_acc) + 1e-9 >= float(min_val_accuracy)
    if sharp_ok and acc_floor_ok:
        first_sharp = best_sharp_loss == float("inf")
        loss_better = float(val_loss) + 1e-9 < float(best_sharp_loss)
        same_loss_better_acc = abs(float(val_loss) - float(best_sharp_loss)) <= 1e-9 and val_acc > best_sharp_acc + 1e-6
        if first_sharp or loss_better or same_loss_better_acc:
            best_sharp_acc = val_acc
            best_sharp_loss = float(val_loss)
            best_sharp_state = (
                best_state
                if best_state is not None
                else {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            )
    improved_any = loss_improved or acc_improved
    return (
        best_val_loss,
        best_val_acc,
        best_sharp_acc,
        best_sharp_loss,
        best_state,
        best_sharp_state,
        improved_any,
    )


def prefer_sharp_checkpoint(
    best_state: dict | None,
    best_sharp_state: dict | None,
) -> dict | None:
    """Prefere o melhor estado que respeitou o piso de sharpness OOS."""
    if best_sharp_state is not None:
        return best_sharp_state
    return best_state
