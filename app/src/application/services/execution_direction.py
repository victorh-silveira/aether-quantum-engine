"""Resolucao e inversao de direcao CALL/PUT para execucao."""

from src.domain.models.trade import TradeDirection
from src.domain.symbols.range_symbols import HEDGE_PEER, hedge_peer, is_high_side


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


def recovery_hedge_target(
    last_loss_symbol: str | None,
    last_loss_direction: str | None,
) -> tuple[str, TradeDirection] | None:
    """Define simbolo par e direcao de hedge apos loss em contratos Range R_*."""
    if not last_loss_symbol or last_loss_symbol not in HEDGE_PEER:
        return None
    if not last_loss_direction:
        return None
    peer = hedge_peer(last_loss_symbol)
    if peer is None:
        return None
    ld = str(last_loss_direction or "").upper()
    if is_high_side(last_loss_symbol):
        hedge_dir = TradeDirection.CALL if ld == "PUT" else TradeDirection.PUT
    else:
        hedge_dir = TradeDirection.PUT if ld == "CALL" else TradeDirection.CALL
    return peer, hedge_dir


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


def build_forced_direction_candidate(
    symbol: str,
    entry: dict,
    forced_dir: TradeDirection,
) -> tuple[str, TradeDirection, dict] | None:
    """Monta candidato com direcao de hedge forcada para recovery no par Range."""
    dl_dir = infer_dl_direction(entry)
    if dl_dir is None:
        return None
    metrics = dict(entry.get("metrics") or {})
    metrics["dl_direction"] = dl_dir.name
    metrics["exec_direction"] = forced_dir.name
    metrics["direction_inverted"] = dl_dir != forced_dir
    metrics["recovery_hedge_forced"] = True
    metrics["execute"] = True
    return symbol, forced_dir, metrics
