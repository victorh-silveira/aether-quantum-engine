"""Simbolos de trading Deriv (universo single-symbol OTC_SPC)."""

TRADING_SYMBOLS: tuple[str, ...] = ("OTC_SPC",)
DRIFT_SYMBOLS: tuple[str, ...] = TRADING_SYMBOLS
DEFAULT_ANCHOR = "OTC_SPC"

HEDGE_PEER: dict[str, str] = {}

HIGH_SIDE: frozenset[str] = frozenset()
LOW_SIDE: frozenset[str] = frozenset()

_SYMBOL_ORDER = {symbol: index for index, symbol in enumerate(TRADING_SYMBOLS)}


def hedge_peer(symbol: str) -> str | None:
    """Retorna o simbolo par de hedge ou None quando nao ha par configurado."""
    return HEDGE_PEER.get(str(symbol))


def is_high_side(symbol: str) -> bool:
    """True quando o simbolo esta no lado high do universo (vazio em OTC_SPC)."""
    return str(symbol) in HIGH_SIDE


def sym_is_low_barrier(symbol: str, peer: str | None = None) -> bool:
    """True quando o simbolo e low-barrier relativo ao peer (sempre False sem peer)."""
    peer_key = peer if peer is not None else hedge_peer(symbol)
    if peer_key is None:
        return False
    return _SYMBOL_ORDER.get(str(symbol), 0) < _SYMBOL_ORDER.get(str(peer_key), 0)
