"""Simbolos de trading Deriv (universo single-symbol OTC_SSMI e rotacao OTC)."""

import time
from datetime import datetime, timedelta, timezone


TRADING_SYMBOLS: tuple[str, ...] = ("OTC_SSMI",)
DRIFT_SYMBOLS: tuple[str, ...] = TRADING_SYMBOLS
DEFAULT_ANCHOR = "OTC_SSMI"

EUROPEAN_OTC_SYMBOLS: frozenset[str] = frozenset(
    {"OTC_SSMI", "OTC_SX5E", "OTC_GDAXI", "OTC_FCHI", "OTC_AEX", "OTC_FTSE"}
)
US_OTC_SYMBOLS: frozenset[str] = frozenset({"OTC_SPC", "OTC_NDX", "OTC_DJI"})
ASIAN_OTC_SYMBOLS: frozenset[str] = frozenset({"OTC_N225", "OTC_HSI", "OTC_AS51"})
_BRT_TZ = timezone(timedelta(hours=-3))

HEDGE_PEER: dict[str, str] = {}

HIGH_SIDE: frozenset[str] = frozenset()
LOW_SIDE: frozenset[str] = frozenset()

_SYMBOL_ORDER = {symbol: index for index, symbol in enumerate(TRADING_SYMBOLS)}


def is_otc_exchange_open_brt(symbol: str, epoch_ts: float | None = None) -> bool:
    """Retorna True se o pregao base do indice OTC estiver aberto no horario de Brasilia."""
    sym = str(symbol).upper()
    if not sym.startswith("OTC_"):
        return True
    ts = float(epoch_ts) if epoch_ts is not None else time.time()
    dt = datetime.fromtimestamp(ts, tz=_BRT_TZ)
    if dt.weekday() >= 5:
        return False
    minutes = dt.hour * 60 + dt.minute
    if sym in EUROPEAN_OTC_SYMBOLS:
        return 180 <= minutes <= 810
    if sym in US_OTC_SYMBOLS:
        return 600 <= minutes <= 1080
    if sym in ASIAN_OTC_SYMBOLS:
        return minutes >= 1200 or minutes <= 300
    return True


def hedge_peer(symbol: str) -> str | None:
    """Retorna o simbolo par de hedge ou None quando nao ha par configurado."""
    return HEDGE_PEER.get(str(symbol))


def is_high_side(symbol: str) -> bool:
    """True quando o simbolo esta no lado high do universo (vazio em single-symbol)."""
    return str(symbol) in HIGH_SIDE


def sym_is_low_barrier(symbol: str, peer: str | None = None) -> bool:
    """True quando o simbolo e low-barrier relativo ao peer (sempre False sem peer)."""
    peer_key = peer if peer is not None else hedge_peer(symbol)
    if peer_key is None:
        return False
    return _SYMBOL_ORDER.get(str(symbol), 0) < _SYMBOL_ORDER.get(str(peer_key), 0)
