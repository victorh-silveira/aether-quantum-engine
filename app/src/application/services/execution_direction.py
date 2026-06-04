"""Resolucao e inversao de direcao CALL/PUT para execucao."""

from src.domain.models.trade import TradeDirection


def infer_dl_direction(entry: dict) -> TradeDirection | None:
    """Obtem direcao prevista pelo DL ou infere a partir de raw_prob."""
    direction = entry.get("direction")
    if direction is not None:
        return direction
    metrics = entry.get("metrics") or {}
    raw = metrics.get("raw_prob")
    if raw is None:
        return None
    return TradeDirection.CALL if float(raw) > 0.5 else TradeDirection.PUT


def invert_direction(direction: TradeDirection) -> TradeDirection:
    """Inverte CALL para PUT e vice-versa."""
    if direction == TradeDirection.CALL:
        return TradeDirection.PUT
    return TradeDirection.CALL


def build_execution_candidate(
    symbol: str,
    entry: dict,
    *,
    invert_dl_direction: bool,
) -> tuple[str, TradeDirection, dict] | None:
    """Monta candidato de ordem com metricas de direcao DL e de execucao."""
    dl_dir = infer_dl_direction(entry)
    if dl_dir is None:
        return None
    exec_dir = invert_direction(dl_dir) if invert_dl_direction else dl_dir
    metrics = dict(entry.get("metrics") or {})
    metrics["dl_direction"] = dl_dir.name
    metrics["exec_direction"] = exec_dir.name
    metrics["direction_inverted"] = bool(invert_dl_direction)
    return symbol, exec_dir, metrics
