from __future__ import annotations


def should_retrain_after_learn(
    *,
    label: str,
    buffer_n: int,
    retrain_min_n: int,
    retrain_on_loss_min_n: int = 2,
) -> bool:
    n = int(buffer_n)
    label_u = str(label).strip().upper()
    if label_u == "LOSS":
        return n >= int(retrain_on_loss_min_n)
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
