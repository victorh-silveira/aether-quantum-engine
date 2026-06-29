"""Normalizacao de bias lateral para squeeze gate."""

_TEXT_CALL = frozenset({"CALL", "RISE", "UP", "BULL"})
_TEXT_PUT = frozenset({"PUT", "FALL", "DOWN", "BEAR"})
_SIDE_MAP = dict.fromkeys(_TEXT_CALL, "call") | dict.fromkeys(_TEXT_PUT, "put")


def normalize_bias_side(value: float | str | None) -> str | None:
    """Converte bias textual ou numerico em call/put."""
    if value is None:
        return None
    if isinstance(value, str):
        return _SIDE_MAP.get(value.strip().upper())
    num = float(value)
    if num > 0.52:
        return "call"
    if num < 0.48:
        return "put"
    return None
