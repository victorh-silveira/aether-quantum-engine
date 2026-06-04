"""Veto de reentrada na mesma combinacao simbolo+direcao apos loss."""

from src.domain.models.trade import TradeDirection


def _ban_list(orch) -> list[dict]:
    """Retorna lista mutavel de vetos pos-loss no orquestrador."""
    bans = getattr(orch, "_dl_post_loss_bans", None)
    if bans is None:
        orch._dl_post_loss_bans = []
        return orch._dl_post_loss_bans
    return bans


def register_post_loss_ban(
    orch,
    symbol: str,
    direction: TradeDirection,
    *,
    candle_cycles: int = 3,
) -> None:
    """Registra veto temporario para mesma direcao no simbolo apos loss."""
    if candle_cycles <= 0:
        return
    bans = _ban_list(orch)
    key_sym = str(symbol)
    key_dir = direction.name
    bans[:] = [b for b in bans if not (b["symbol"] == key_sym and b["direction"] == key_dir)]
    bans.append({"symbol": key_sym, "direction": key_dir, "remaining": int(candle_cycles)})


def tick_post_loss_bans(orch) -> None:
    """Decrementa vetos pos-loss a cada nova vela do ancora."""
    bans = getattr(orch, "_dl_post_loss_bans", None)
    if not bans:
        return
    for entry in bans:
        entry["remaining"] = int(entry["remaining"]) - 1
    orch._dl_post_loss_bans = [b for b in bans if int(b["remaining"]) > 0]


def post_loss_block_reason(
    orch,
    symbol: str,
    direction: TradeDirection | None,
    *,
    raw_prob: float | None = None,
    flip_raw_min: float = 0.58,
) -> str | None:
    """Retorna repeat_loss se combinacao ainda vetada e raw nao indica flip forte."""
    if direction is None:
        return None
    bans = getattr(orch, "_dl_post_loss_bans", None) or []
    key_sym = str(symbol)
    key_dir = direction.name
    active = any(b["symbol"] == key_sym and b["direction"] == key_dir and int(b["remaining"]) > 0 for b in bans)
    if not active:
        return None
    if raw_prob is not None:
        p = float(raw_prob)
        if direction == TradeDirection.CALL and p + 1e-9 <= 1.0 - float(flip_raw_min):
            return None
        if direction == TradeDirection.PUT and p + 1e-9 >= float(flip_raw_min):
            return None
    return "repeat_loss"


def last_post_loss_pair(orch) -> tuple[str | None, str | None]:
    """Retorna ultimo par simbolo+direcao com loss registrado no orquestrador."""
    sym = getattr(orch, "_last_loss_symbol", None) or getattr(
        getattr(orch, "risk_manager", None), "last_loss_symbol", None
    )
    direction = getattr(orch, "_last_loss_direction", None) or getattr(
        getattr(orch, "risk_manager", None), "last_loss_direction", None
    )
    if sym and not str(sym).strip():
        sym = None
    if direction and not str(direction).strip():
        direction = None
    return (str(sym) if sym else None, str(direction) if direction else None)
