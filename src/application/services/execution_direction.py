"""Resolucao de direcao CALL/PUT para execucao (sem inversao)."""

from typing import Any

from src.application.services.execution_direction_resolver import (
    infer_dl_direction,
    is_technically_blocked,
    resolve_execution_direction,
)
from src.application.services.execution_runtime_config import resolve_cross_corr_config
from src.domain.models.trade import TradeDirection
from src.domain.risk.stake_sizing import enrich_metrics_conviction, metric_float, raw_side_from_metrics
from src.domain.symbols.drift_symbols import hedge_peer


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


def mandatory_execution_eligible(entry: dict, *, min_signal: float = 0.56, min_val_accuracy: float = 0.50) -> bool:
    """Indica se fallback obrigatorio pode operar com sinal minimo."""
    if is_technically_blocked(entry):
        return False
    if infer_dl_direction(entry) is None:
        return False
    return meets_mandatory_signal_floor(entry.get("metrics") or {}, min_signal=min_signal, min_val=min_val_accuracy)


def recovery_execution_eligible(entry: dict) -> bool:
    """Indica se candidato tem dados tecnicos validos para recovery."""
    if is_technically_blocked(entry):
        return False
    return infer_dl_direction(entry) is not None


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
    skipped_cycles_counter: int | None = None,
    orch: Any | None = None,
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
        skipped_cycles_counter=skipped_cycles_counter,
        orch=orch,
    )
    if resolved is None:
        return None
    direction, metrics = resolved
    enrich_metrics_conviction(metrics)
    return symbol, direction, metrics


def _entry_gate_blocked(metrics: dict) -> bool:
    """Indica bloqueio absoluto para fallback obrigatorio de execucao."""
    if metrics.get("deploy_ok") is False:
        return True
    gate = str(metrics.get("gate_reason") or "")
    return gate in _TECHNICAL_BLOCKS


def _squeeze_active(metrics: dict) -> bool:
    """Indica regime de squeeze/baixa volatilidade para peso de correlacao."""
    indicators = metrics.get("indicators") or {}
    vol_ratio = float(indicators["vol_ratio"]) if "vol_ratio" in indicators else 1.0
    if vol_ratio < float(resolve_cross_corr_config()["squeeze_vol_ratio_max"]):
        return True
    return bool(metrics.get("bb_squeeze") or metrics.get("squeeze_active"))


def _weak_consensus(metrics: dict, *, min_margin: float) -> bool:
    """Indica consenso direcional fraco."""
    margin = float(metrics.get("direction_margin", 0.0))
    return margin + 1e-9 < float(min_margin)


def adjust_dl_weight_with_correlation(
    weights: dict[str, float],
    symbol: str,
    metrics: dict,
    corr_matrix: dict[tuple[str, str], float],
    *,
    anchor: str = "1HZ75V",
    min_margin: float | None = None,
) -> dict[str, float]:
    """Ajusta peso DL dinamicamente via correlacao cruzada entre indices."""
    cfg = resolve_cross_corr_config()
    margin_floor = float(cfg["min_margin"]) if min_margin is None else float(min_margin)
    if not _squeeze_active(metrics) and not _weak_consensus(metrics, min_margin=margin_floor):
        return weights
    merged = dict(weights)
    base = float(merged["dl_raw_weight"]) if "dl_raw_weight" in merged else float(cfg["dl_raw_weight"])
    corr = float(corr_matrix.get((str(symbol), str(anchor)), corr_matrix.get((str(anchor), str(symbol)), 0.0)))
    corr = max(-1.0, min(1.0, corr))
    if abs(corr) > float(cfg["high_corr_abs"]):
        retention = max(
            float(cfg["high_corr_retention_floor"]), 1.0 - abs(corr) * float(cfg["high_corr_retention_coef"])
        )
        merged["dl_raw_weight"] = base * retention
        metrics["cross_corr_dl_retention"] = retention
        metrics["cross_corr_anchor"] = anchor
    elif abs(corr) < float(cfg["low_corr_abs"]):
        merged["dl_raw_weight"] = min(float(cfg["low_corr_weight_cap"]), base * float(cfg["low_corr_weight_boost"]))
    return merged


def cached_correlation_matrix(orch: Any) -> dict[tuple[str, str], float]:
    """Retorna matriz de correlacao do cache local do orquestrador."""
    cached = getattr(orch, "_corr_matrix_cache", None)
    if isinstance(cached, dict):
        return cached
    return {}
