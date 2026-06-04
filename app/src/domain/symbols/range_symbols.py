"""Símbolos Range R_* da Deriv, pares de hedge e ordem de barreira."""

RANGE_SYMBOLS: tuple[str, ...] = ("R_10", "R_25", "R_50", "R_75", "R_100")
DEFAULT_ANCHOR = "R_50"

HEDGE_PEER: dict[str, str] = {
    "R_10": "R_100",
    "R_100": "R_10",
    "R_25": "R_75",
    "R_75": "R_25",
}

HIGH_SIDE = frozenset({"R_75", "R_100"})
LOW_SIDE = frozenset({"R_10", "R_25"})

_SYMBOL_ORDER = {symbol: index for index, symbol in enumerate(RANGE_SYMBOLS)}


def hedge_peer(symbol: str) -> str | None:
    """Retorna o símbolo par de hedge ou None quando não há par configurado."""
    return HEDGE_PEER.get(str(symbol))


def is_high_side(symbol: str) -> bool:
    """True para barreiras altas (R_75, R_100) usadas na lógica de recovery."""
    return str(symbol) in HIGH_SIDE


def sym_is_low_barrier(symbol: str, peer: str | None = None) -> bool:
    """True quando o símbolo tem barreira menor que o par no eixo R_*."""
    peer_key = peer if peer is not None else hedge_peer(symbol)
    if peer_key is None:
        return False
    return _SYMBOL_ORDER.get(str(symbol), 0) < _SYMBOL_ORDER.get(str(peer_key), 0)
