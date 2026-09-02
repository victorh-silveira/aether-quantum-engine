from __future__ import annotations


def bootstrap_retrain_floor(
    *,
    retrain_on_loss_min_n: int = 2,
    bootstrap_exit_n: int = 16,
) -> int:
    return int(min(int(bootstrap_exit_n), max(4, int(retrain_on_loss_min_n))))


def should_retrain_after_learn(
    *,
    label: str,
    buffer_n: int,
    retrain_min_n: int,
    retrain_on_loss_min_n: int = 2,
    buffer_win: int = 0,
    buffer_loss: int = 0,
    max_loss_frac: float = 0.60,
    min_win_for_loss_retrain: int = 4,
    bootstrap_active: bool = False,
    bootstrap_exit_n: int = 16,
) -> bool:
    n = int(buffer_n)
    wins = int(buffer_win)
    losses = int(buffer_loss)
    label_u = str(label).strip().upper()
    min_n = int(retrain_min_n)
    if wins < 1 or losses < 1:
        return False
    if bool(bootstrap_active):
        floor = bootstrap_retrain_floor(
            retrain_on_loss_min_n=int(retrain_on_loss_min_n),
            bootstrap_exit_n=int(bootstrap_exit_n),
        )
        return n >= floor
    if label_u == "LOSS":
        if n < int(retrain_on_loss_min_n):
            return False
        if wins < int(min_win_for_loss_retrain):
            return False
        denom = max(n, 1)
        if float(losses) / float(denom) > float(max_loss_frac):
            return False
        return True
    if n < max(2, int(min_n)):
        return False
    return True


def retrain_min_for_label(
    *,
    label: str,
    retrain_min_n: int,
    retrain_on_loss_min_n: int = 2,
    bootstrap_active: bool = False,
    bootstrap_exit_n: int = 16,
) -> int:
    if bool(bootstrap_active):
        return bootstrap_retrain_floor(
            retrain_on_loss_min_n=int(retrain_on_loss_min_n),
            bootstrap_exit_n=int(bootstrap_exit_n),
        )
    if str(label).strip().upper() == "LOSS":
        return int(retrain_on_loss_min_n)
    return int(retrain_min_n)


def retrain_skipped_reason(
    *,
    label: str,
    buffer_n: int,
    retrain_min_n: int,
    retrain_on_loss_min_n: int = 2,
    buffer_win: int = 0,
    buffer_loss: int = 0,
    max_loss_frac: float = 0.60,
    min_win_for_loss_retrain: int = 4,
    bootstrap_active: bool = False,
    bootstrap_exit_n: int = 16,
    should_retrain: bool | None = None,
) -> str:
    if should_retrain is None:
        should_retrain = should_retrain_after_learn(
            label=label,
            buffer_n=buffer_n,
            retrain_min_n=retrain_min_n,
            retrain_on_loss_min_n=retrain_on_loss_min_n,
            buffer_win=buffer_win,
            buffer_loss=buffer_loss,
            max_loss_frac=max_loss_frac,
            min_win_for_loss_retrain=min_win_for_loss_retrain,
            bootstrap_active=bootstrap_active,
            bootstrap_exit_n=bootstrap_exit_n,
        )
    if should_retrain:
        return "ok"
    n = int(buffer_n)
    wins = int(buffer_win)
    losses = int(buffer_loss)
    label_u = str(label).strip().upper()
    if bool(bootstrap_active):
        floor = bootstrap_retrain_floor(
            retrain_on_loss_min_n=int(retrain_on_loss_min_n),
            bootstrap_exit_n=int(bootstrap_exit_n),
        )
        if wins < 1 or losses < 1:
            return f"bootstrap_need_classes:w{wins}/l{losses}"
        return f"bootstrap_wait:{n}/{floor}"
    if wins < 1 or losses < 1:
        return f"need_classes:w{wins}/l{losses}"
    if label_u == "LOSS":
        if n < int(retrain_on_loss_min_n):
            return f"need_n:{n}/{int(retrain_on_loss_min_n)}"
        if wins < int(min_win_for_loss_retrain):
            return f"need_wins:{wins}/{int(min_win_for_loss_retrain)}"
        denom = max(n, 1)
        if float(losses) / float(denom) > float(max_loss_frac):
            return f"loss_frac:{losses}/{n}"
        return "skip"
    if n < max(2, int(retrain_min_n)):
        return f"need_n:{n}/{max(2, int(retrain_min_n))}"
    return "skip"
