"""Normalização de lista de símbolos e âncora a partir do JSON de configuração."""


def normalize_symbols_and_anchor(config: dict) -> tuple[str, list[str]]:
    """Deriva âncora e símbolos ativos (symbols menos excluded_symbols)."""
    anchor = str(config.get("anchor", "stp_500"))
    strategy = config.get("strategy", {})
    raw = config.get("symbols") or [anchor]
    excluded = {str(x) for x in ((strategy or {}).get("excluded_symbols") or [])}
    symbols = [s for s in dict.fromkeys(raw) if s not in excluded or s == anchor]
    if anchor not in symbols:
        symbols.insert(0, anchor)
    return anchor, symbols


def resolve_dl_train_symbols(config: dict) -> list[str]:
    """Retorna simbolos que o treino DL pode persistir; padrao e symbols do motor."""
    dl_cfg = config.get("deep_learning") if isinstance(config.get("deep_learning"), dict) else {}
    raw = dl_cfg.get("train_symbols")
    if isinstance(raw, list):
        return [str(symbol) for symbol in dict.fromkeys(raw)]
    _, symbols = normalize_symbols_and_anchor(config)
    return symbols
