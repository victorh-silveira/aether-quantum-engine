"""Martingale em recovery: modo nativo sem bloqueios por metricas DL."""


def martingale_pending_total(pending_loss: dict[str, float]) -> float:
    """Soma perdas pendentes de todos os simbolos."""
    return sum(float(v) for v in pending_loss.values())


def martingale_allowed(*, pending_loss: dict[str, float]) -> bool:
    """Indica se martingale pode ser usado neste ciclo de recovery."""
    return martingale_pending_total(pending_loss) > 0.0


def apply_win_to_pending_loss(pending_loss: dict[str, float], profit: float) -> None:
    """Reduz perdas pendentes com lucro parcial de um contrato."""
    remaining_profit = profit
    for sym in list(pending_loss.keys()):
        if remaining_profit <= 0:
            break
        current_loss = pending_loss[sym]
        if current_loss <= remaining_profit:
            remaining_profit -= current_loss
            pending_loss[sym] = 0.0
        else:
            pending_loss[sym] = current_loss - remaining_profit
            remaining_profit = 0.0
