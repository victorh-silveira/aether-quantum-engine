"""Resolucao de timeframe e granularidade de treino DL."""


def resolve_train_timeframe(dl_config: dict | None = None) -> str:
    """Normaliza train_timeframe para macro ou micro."""
    raw = str((dl_config or {}).get("train_timeframe", "macro")).strip().lower()
    if raw in ("micro", "m5", "cycle", "settlement"):
        return "micro"
    return "macro"


def resolve_dl_granularity(dl_config: dict, data_config: dict | None = None) -> int:
    """Escolhe granularidade de treino conforme timeframe macro/micro."""
    data_config = data_config or {}
    macro = int(data_config.get("granularity") or dl_config.get("granularity") or 60)
    micro = int(data_config.get("micro_granularity") or dl_config.get("micro_granularity") or macro)
    if resolve_train_timeframe(dl_config) == "micro":
        return max(1, micro)
    return max(1, macro)
