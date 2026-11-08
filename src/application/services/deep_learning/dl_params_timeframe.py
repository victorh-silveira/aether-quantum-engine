"""Resolucao de timeframe e granularidade de treino DL."""


def resolve_train_timeframe(dl_config: dict | None = None) -> str:
    """Normaliza train_timeframe para micro ou macro."""
    raw = str((dl_config or {}).get("train_timeframe", "micro")).strip().lower()
    if raw in ("macro", "d1"):
        return "macro"
    return "micro"


def resolve_dl_granularity(dl_config: dict, data_config: dict | None = None) -> int:
    """Escolhe granularidade de treino conforme timeframe macro/micro."""
    data_config = data_config or {}
    macro = int(data_config.get("granularity") or dl_config.get("granularity") or 3600)
    micro = int(
        data_config.get("micro_granularity")
        or dl_config.get("micro_granularity")
        or data_config.get("granularity")
        or 180
    )
    if resolve_train_timeframe(dl_config) == "micro":
        return max(1, micro)
    return max(1, macro)
