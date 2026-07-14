"""Resolucao e inversao de direcao CALL/PUT para execucao."""

from src.application.services.execution_direction_resolver import (
    infer_dl_direction,
    is_technically_blocked,
    resolve_execution_direction,
)
from src.domain.models.trade import TradeDirection
from src.domain.risk.stake_sizing import enrich_metrics_conviction, metric_float, raw_side_from_metrics
from src.domain.symbols.drift_symbols import HEDGE_PEER, hedge_peer, is_high_side


_TECHNICAL_BLOCKS = frozenset({"data", "predict_error", "training"})


def _entry_signal_strength(metrics: dict) -> tuple[float, float]:
    """Extrai score calibrado e conviccao bruta lateralizada do candidato."""
    score = metric_float(metrics, "trade_score", "conviction", default=0.0)
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
    """Indica se fallback obrigatorio pode operar com sinal minimo."""
    if is_technically_blocked(entry):
        return False
    if infer_dl_direction(entry) is None:
        return False
    return meets_mandatory_signal_floor(
        entry.get("metrics") or {},
        min_signal=min_signal,
        min_val=min_val_accuracy,
    )


def recovery_execution_eligible(entry: dict, recovery_cfg: dict | None = None) -> bool:
    """Indica se candidato tem dados tecnicos validos para recovery."""
    _ = recovery_cfg
    if is_technically_blocked(entry):
        return False
    return infer_dl_direction(entry) is not None


def recovery_hedge_target(
    last_loss_symbol: str | None,
    last_loss_direction: str | None,
) -> tuple[str, TradeDirection] | None:
    """Define simbolo par e direcao de hedge apos loss em contratos Drift."""
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
    exec_cfg: dict | None = None,
    calibration_cfg: dict | None = None,
    recovery_active: bool = False,
    corr_matrix: dict[tuple[str, str], float] | None = None,
    infra_cfg: dict | None = None,
    decisions: dict | None = None,
    cycle_id: int = 0,
    risk_manager=None,
) -> tuple[str, TradeDirection, dict] | None:
    """Monta candidato com direcao resolvida por scoring inteligente."""
    peer = hedge_peer(symbol)
    peer_entry = None
    if isinstance(decisions, dict) and peer:
        peer_entry = decisions.get(peer)
    resolved = resolve_execution_direction(
        entry,
        exec_cfg=exec_cfg or {},
        calibration_cfg=calibration_cfg,
        recovery_active=recovery_active,
        symbol=symbol,
        corr_matrix=corr_matrix,
        infra_cfg=infra_cfg,
        peer_entry=peer_entry if isinstance(peer_entry, dict) else None,
        cycle_id=cycle_id,
        risk_manager=risk_manager,
    )
    if resolved is None:
        return None
    direction, metrics = resolved
    enrich_metrics_conviction(metrics)
    return symbol, direction, metrics


def build_forced_direction_candidate(
    symbol: str,
    entry: dict,
    forced_dir: TradeDirection,
) -> tuple[str, TradeDirection, dict] | None:
    """Monta candidato com direcao de hedge forcada para recovery no par Drift."""
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
    return gate in _TECHNICAL_BLOCKS
