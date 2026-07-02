"""Simbolos Drift RDBEAR/RDBULL da Deriv e par de hedge."""

DRIFT_SYMBOLS: tuple[str, ...] = ("RDBEAR", "RDBULL")
DEFAULT_ANCHOR = "RDBULL"

HEDGE_PEER: dict[str, str] = {
    "RDBEAR": "RDBULL",
    "RDBULL": "RDBEAR",
}

HIGH_SIDE = frozenset({"RDBULL"})
LOW_SIDE = frozenset({"RDBEAR"})

_SYMBOL_ORDER = {symbol: index for index, symbol in enumerate(DRIFT_SYMBOLS)}


def hedge_peer(symbol: str) -> str | None:
    """Retorna o simbolo par de hedge ou None quando nao ha par configurado."""
    return HEDGE_PEER.get(str(symbol))


def is_high_side(symbol: str) -> bool:
    """True para RDBULL usado na logica de recovery."""
    return str(symbol) in HIGH_SIDE


def sym_is_low_barrier(symbol: str, peer: str | None = None) -> bool:
    """True quando o simbolo e RDBEAR em relacao ao par Drift."""
    peer_key = peer if peer is not None else hedge_peer(symbol)
    if peer_key is None:
        return False
    return _SYMBOL_ORDER.get(str(symbol), 0) < _SYMBOL_ORDER.get(str(peer_key), 0)
