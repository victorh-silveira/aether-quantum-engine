"""Normalizacao de lista de simbolos e ancora a partir do JSON de configuracao."""

from src.application.services.llm.strategy_clusters import resolve_cluster_lists


def normalize_symbols_and_anchor(config: dict) -> tuple[str, list[str]]:
    """Deriva ancora e simbolos ativos (clusters menos excluded_symbols)."""
    anchor = str(config.get("anchor", "frxEURUSD"))
    strategy = config.get("strategy", {})
    us, eu = resolve_cluster_lists(strategy if isinstance(strategy, dict) else None)
    if us or eu:
        symbols = list(dict.fromkeys([anchor, *us, *eu]))
        return anchor, symbols
    raw = config.get("symbols") or [anchor]
    excluded = {str(x) for x in ((strategy or {}).get("excluded_symbols") or [])}
    symbols = [s for s in dict.fromkeys(raw) if s not in excluded or s == anchor]
    if anchor not in symbols:
        symbols.insert(0, anchor)
    return anchor, symbols
