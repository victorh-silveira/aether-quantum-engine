from __future__ import annotations


def should_retrain_after_learn(
    *,
    label: str,
    buffer_n: int,
    retrain_min_n: int,
    retrain_on_loss_min_n: int = 2,
    buffer_win: int = 0,
    buffer_loss: int = 0,
    max_loss_frac: float = 0.60,
    min_win_for_loss_retrain: int = 8,
    bootstrap_active: bool = False,
) -> bool:
    n = int(buffer_n)
    wins = int(buffer_win)
    losses = int(buffer_loss)
    label_u = str(label).strip().upper()
    if bool(bootstrap_active) and wins >= 1 and losses >= 1 and n >= int(retrain_on_loss_min_n):
        return True
    if label_u == "LOSS":
        if n < int(retrain_on_loss_min_n):
            return False
        if wins < int(min_win_for_loss_retrain):
            return False
        denom = max(n, 1)
        if float(losses) / float(denom) > float(max_loss_frac):
            return False
        return True
    if n < int(retrain_min_n):
        return False
    step = max(8, int(retrain_min_n) // 4)
    return n % step == 0


def retrain_min_for_label(
    *,
    label: str,
    retrain_min_n: int,
    retrain_on_loss_min_n: int = 2,
) -> int:
    if str(label).strip().upper() == "LOSS":
        return int(retrain_on_loss_min_n)
    return int(retrain_min_n)
