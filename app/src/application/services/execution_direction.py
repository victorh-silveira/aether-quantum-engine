"""Resolucao e inversao de direcao CALL/PUT para execucao."""

from src.domain.models.trade import TradeDirection
from src.domain.risk.stake_sizing import enrich_metrics_conviction, raw_side_from_metrics
from src.domain.symbols.range_symbols import HEDGE_PEER, hedge_peer, is_high_side


_MANDATORY_HARD_BLOCKS = frozenset({"trend_conflict"})
_MANDATORY_POOL_HARD_BLOCKS = frozenset({"trend_conflict"})
_FORCED_ENTRY_HARD_BLOCKS = frozenset({"trend_conflict"})


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


def _entry_signal_strength(metrics: dict) -> tuple[float, float]:
    """Extrai score calibrado e conviccao bruta lateralizada do candidato."""
    score = float(metrics.get("trade_score", metrics.get("conviction", 0.0)))
    raw_side = raw_side_from_metrics(metrics)
    return score, raw_side


def meets_mandatory_signal_floor(metrics: dict, *, min_signal: float, min_val: float) -> bool:
    """Verifica se metricas atendem piso minimo de sinal e val_accuracy."""
    score, raw_side = _entry_signal_strength(metrics)
    if max(score, raw_side) + 1e-9 < min_signal:
        return False
    return not (min_val > 0.0 and float(metrics.get("val_accuracy", 0.0)) + 1e-9 < min_val)


def mandatory_execution_eligible(
    entry: dict,
    *,
    min_signal: float = 0.56,
    min_val_accuracy: float = 0.50,
) -> bool:
    """Indica se o modo obrigatorio pode operar apesar de execute=false no gating DL."""
    metrics = entry.get("metrics") or {}
    if metrics.get("deploy_ok") is False:
        return False
    if infer_dl_direction(entry) is None:
        return False
    return meets_mandatory_signal_floor(metrics, min_signal=min_signal, min_val=min_val_accuracy)


def recovery_execution_eligible(entry: dict, recovery_cfg: dict | None = None) -> bool:
    """Indica se candidato tem qualidade minima para recovery sem martingale cego."""
    metrics = entry.get("metrics") or {}
    if metrics.get("execute", False):
        return True
    if metrics.get("deploy_ok") is False:
        return False
    cfg = recovery_cfg or {}
    gate = str(metrics.get("gate_reason") or "")
    score, raw_side = _entry_signal_strength(metrics)
    val_acc_bypass = False
    if gate == "val_acc":
        bypass_raw = float(cfg.get("val_acc_bypass_min_raw", 0.0))
        if bypass_raw > 0.0 and raw_side + 1e-9 >= bypass_raw:
            gate = ""
            val_acc_bypass = True
    if infer_dl_direction(entry) is None:
        return False
    min_conv = float(cfg.get("min_conviction_execute", 0.53))
    min_val = float(cfg.get("min_val_accuracy", 0.50))
    val = float(metrics.get("val_accuracy", 0.0))
    min_raw = float(cfg.get("min_raw_conviction_execute", 0.55))
    val_ok = val_acc_bypass or val + 1e-9 >= min_val
    return score + 1e-9 >= min_conv and val_ok and (raw_side + 1e-9 >= min_raw or score + 1e-9 >= min_conv)


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
) -> tuple[str, TradeDirection, dict] | None:
    """Monta candidato de ordem com metricas de direcao DL e de execucao."""
    dl_dir = infer_dl_direction(entry)
    if dl_dir is None:
        return None
    metrics = dict(entry.get("metrics") or {})
    metrics["dl_direction"] = dl_dir.name
    metrics["exec_direction"] = dl_dir.name
    metrics["direction_inverted"] = False
    return symbol, dl_dir, metrics


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
    return symbol, forced_dir, metrics


def build_forced_recovery_candidate(
    symbol: str,
    entry: dict,
    forced_dir: TradeDirection,
) -> tuple[str, TradeDirection, dict]:
    """Monta candidato de recovery com direcao forcada igual ao ultimo loss."""
    metrics = dict(entry.get("metrics") or {})
    dl_dir = infer_dl_direction(entry)
    metrics["dl_direction"] = dl_dir.name if dl_dir else forced_dir.name
    metrics["exec_direction"] = forced_dir.name
    metrics["recovery_forced"] = True
    metrics["direction_inverted"] = dl_dir is not None and dl_dir != forced_dir
    enrich_metrics_conviction(metrics)
    return symbol, forced_dir, metrics


def _entry_gate_blocked(metrics: dict) -> bool:
    """Indica bloqueio absoluto para fallback obrigatorio de execucao."""
    if metrics.get("deploy_ok") is False:
        return True
    gate = str(metrics.get("gate_reason") or "")
    return gate in _FORCED_ENTRY_HARD_BLOCKS
