"""Martingale em recovery: modo nativo sem bloqueios por metricas DL."""

from src.domain.symbols.range_symbols import HEDGE_PEER


def martingale_pending_total(pending_loss: dict[str, float]) -> float:
    """Soma perdas pendentes de todos os simbolos."""
    return sum(float(v) for v in pending_loss.values())


def martingale_repeat_loss_blocked(
    symbol: str,
    order_direction: str | None,
    last_loss_symbol: str | None,
    last_loss_direction: str | None,
) -> bool:
    """True se martingale repete symbol+direction da ultima loss ou o mesmo lado no par."""
    if not order_direction or not last_loss_direction:
        return False
    order = str(order_direction)
    last_dir = str(last_loss_direction)
    if last_loss_symbol == symbol and order == last_dir:
        return True
    if last_loss_symbol in HEDGE_PEER and symbol == HEDGE_PEER.get(last_loss_symbol):
        return order == last_dir
    return False


def martingale_block_reason(
    *,
    pending_loss: dict[str, float],
    martingale_native: bool = True,
    block_repeat_loss: bool = False,
    symbol: str = "",
    order_direction: str | None = None,
    last_loss_symbol: str | None = None,
    last_loss_direction: str | None = None,
    **_legacy: object,
) -> str | None:
    """Retorna motivo de bloqueio do martingale ou None se permitido."""
    loss_to_recover = martingale_pending_total(pending_loss)
    if loss_to_recover <= 0.0:
        return "no_pending_loss"
    if martingale_native:
        if block_repeat_loss and martingale_repeat_loss_blocked(
            symbol, order_direction, last_loss_symbol, last_loss_direction
        ):
            return "repeat_loss"
        return None
    return "legacy_disabled"


def martingale_allowed(
    *,
    pending_loss: dict[str, float],
    martingale_native: bool = True,
    block_repeat_loss: bool = False,
    symbol: str = "",
    order_direction: str | None = None,
    last_loss_symbol: str | None = None,
    last_loss_direction: str | None = None,
    **_legacy: object,
) -> bool:
    """Indica se martingale pode ser usado neste ciclo de recovery."""
    return (
        martingale_block_reason(
            pending_loss=pending_loss,
            martingale_native=martingale_native,
            block_repeat_loss=block_repeat_loss,
            symbol=symbol,
            order_direction=order_direction,
            last_loss_symbol=last_loss_symbol,
            last_loss_direction=last_loss_direction,
        )
        is None
    )


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
