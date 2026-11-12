"""Simbolos de trading Deriv (universo single-symbol OTC_SSMI e rotacao OTC)."""

import time
from datetime import UTC, datetime, timedelta, timezone


TRADING_SYMBOLS: tuple[str, ...] = ("OTC_SSMI",)
DRIFT_SYMBOLS: tuple[str, ...] = TRADING_SYMBOLS
DEFAULT_ANCHOR = "OTC_SSMI"

EUROPEAN_OTC_SYMBOLS: frozenset[str] = frozenset(
    {"OTC_SSMI", "OTC_SX5E", "OTC_GDAXI", "OTC_FCHI", "OTC_AEX", "OTC_FTSE"}
)
US_OTC_SYMBOLS: frozenset[str] = frozenset({"OTC_SPC", "OTC_NDX", "OTC_DJI"})
ASIAN_OTC_SYMBOLS: frozenset[str] = frozenset({"OTC_N225", "OTC_HSI", "OTC_AS51"})
_BRT_TZ = timezone(timedelta(hours=-3))

SYMBOL_DISPLAY_NAMES: dict[str, str] = {
    "OTC_AEX": "Netherlands 25",
    "OTC_AS51": "Australia 200",
    "OTC_DJI": "Wall Street 30",
    "OTC_FCHI": "France 40",
    "OTC_FTSE": "UK 100",
    "OTC_GDAXI": "Germany 40",
    "OTC_HSI": "Hong Kong 50",
    "OTC_N225": "Japan 225",
    "OTC_NDX": "US Tech 100",
    "OTC_SPC": "US 500",
    "OTC_SSMI": "Swiss 20",
    "OTC_SX5E": "Euro 50",
}


def symbol_display_name(symbol: str) -> str:
    """Retorna o nome comercial do ativo na Deriv ou o proprio simbolo."""
    return SYMBOL_DISPLAY_NAMES.get(str(symbol).upper(), str(symbol))


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


def session_liquidity_weight(symbol: str, epoch_ts: float | None = None) -> float:
    """Retorna o peso de liquidez do ativo baseado na sessao de pregao aberta."""
    return 1.0 if is_otc_exchange_open_brt(symbol, epoch_ts) else 0.2


def hedge_peer(symbol: str) -> str | None:
    """Retorna o simbolo par de hedge ou None quando nao ha par configurado."""
    return HEDGE_PEER.get(str(symbol))


def is_high_side(symbol: str) -> bool:
    """True quando o simbolo esta no lado high do universo (vazio em single-symbol)."""
    return str(symbol) in HIGH_SIDE


_BRT_TZ_OFFSET = timezone(timedelta(hours=-3))
_DERIV_OTC_WEEKDAY_HOURS_BRT: dict[str, tuple[int, ...]] = {
    "OTC_SPC": tuple(range(4, 18)),
    "OTC_NDX": tuple(range(4, 18)),
    "OTC_DJI": tuple(range(4, 18)),
    "OTC_SX5E": tuple(range(4, 18)),
    "OTC_FTSE": tuple(range(4, 18)),
    "OTC_FCHI": tuple(range(4, 18)),
    "OTC_GDAXI": tuple(range(4, 18)),
    "OTC_AEX": tuple(range(4, 18)),
    "OTC_SSMI": tuple(range(4, 18)),
    "OTC_AS51": tuple(range(0, 16)) + (20, 21, 22, 23),
    "OTC_N225": tuple(range(0, 17)) + (20, 21, 22, 23),
    "OTC_HSI": (0, 1, 2, 3, 4, 22, 23),
}


def is_deriv_otc_session_open_brt(symbol: str, epoch_ts: float | None = None) -> bool:
    """Informa se a sessao estendida Deriv OTC do ativo opera no horario BRT informado."""
    sym = str(symbol)
    if not sym.startswith("OTC_"):
        return True
    hours = _DERIV_OTC_WEEKDAY_HOURS_BRT.get(sym)
    if hours is None:
        return is_otc_exchange_open_brt(sym, epoch_ts)
    dt = (datetime.fromtimestamp(float(epoch_ts), tz=UTC) if epoch_ts is not None else datetime.now(UTC)).astimezone(
        _BRT_TZ_OFFSET
    )
    wd = dt.weekday()
    if wd == 5:
        return False
    if wd == 6:
        return sym in ("OTC_AS51", "OTC_N225", "OTC_HSI") and dt.hour >= 20
    return dt.hour in hours


def real_market_vs_otc_status(symbol: str, epoch_ts: float | None = None) -> tuple[bool, bool]:
    """Retorna (bolsa_real_aberta, deriv_otc_ativo) em Horario de Brasilia."""
    return is_otc_exchange_open_brt(symbol, epoch_ts), is_deriv_otc_session_open_brt(symbol, epoch_ts)


def sym_is_low_barrier(symbol: str, peer: str | None = None) -> bool:
    """True quando o simbolo e low-barrier relativo ao peer (sempre False sem peer)."""
    peer_key = peer if peer is not None else hedge_peer(symbol)
    if peer_key is None:
        return False
    return _SYMBOL_ORDER.get(str(symbol), 0) < _SYMBOL_ORDER.get(str(peer_key), 0)
