"""Normalização de lista de símbolos e âncora a partir do JSON de configuração."""


def normalize_symbols_and_anchor(config: dict) -> tuple[str, list[str]]:
    """Deriva âncora e símbolos ativos (symbols menos excluded_symbols)."""
    anchor = str(config.get("anchor", "RDBULL"))
    strategy = config.get("strategy", {})
    raw = config.get("symbols") or [anchor]
    excluded = {str(x) for x in ((strategy or {}).get("excluded_symbols") or [])}
    symbols = [s for s in dict.fromkeys(raw) if s not in excluded or s == anchor]
    if anchor not in symbols:
        symbols.insert(0, anchor)
    return anchor, symbols
