"""Normalizacao de lista de simbolos e ancora a partir do JSON de configuracao."""


def normalize_symbols_and_anchor(config: dict) -> tuple[str, list[str]]:
    """Deriva ancora e lista unica de simbolos a partir de symbols ou correlation_engine."""
    ce = config.get("correlation_engine") or {}
    raw = config.get("symbols") or ce.get("symbols") or ["1HZ75V"]
    anchor = config.get("anchor") or ce.get("anchor") or raw[0]
    symbols = list(dict.fromkeys(raw))
    if anchor not in symbols:
        symbols.insert(0, anchor)
    return anchor, symbols
