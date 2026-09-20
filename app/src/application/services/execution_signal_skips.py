"""Gates de SKIP de sinal liberados (ACC, Edge, DOJI, vela, SCALE)."""

from __future__ import annotations

from typing import Any

from src.domain.models.trade import TradeDirection


_VALID = {TradeDirection.CALL.name, TradeDirection.PUT.name}
_DEFAULT_MATERIAL_PEND = 0.5


def _mark_skip(metrics: dict[str, Any], reason: str, **extra: Any) -> None:
    """Marca ciclo como SKIP de sinal (sem EXEC)."""
    metrics["signal_status"] = f"SKIP:{reason}"
    metrics["gate_reason"] = reason
    metrics["skip_reason"] = reason
    metrics["execution_candidate_ready"] = False
    for key, value in extra.items():
        metrics[key] = value


def _closed_candle_dir(metrics: dict[str, Any]) -> str | None:
    """Vela M5 fechada stampada CALL/PUT ou None."""
    if not bool(metrics.get("closed_micro_candle_stamped")):
        return None
    candle = str(metrics.get("closed_micro_candle_dir") or "").strip().upper()
    return candle if candle in _VALID else None


def _material_pending_floor(exec_cfg: dict[str, Any] | None) -> float:
    """Piso de PEND material para waive de scale discord."""
    raw = (exec_cfg or {}).get("material_pending_min")
    if raw is None:
        return _DEFAULT_MATERIAL_PEND
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return _DEFAULT_MATERIAL_PEND


def _pending_loss_total(metrics: dict[str, Any]) -> float:
    """Le PEND stampado nas metrics."""
    raw = metrics.get("pending_loss_total")
    if raw is None:
        return 0.0
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 0.0


def _pend_waives(metrics: dict[str, Any], exec_cfg: dict[str, Any] | None) -> bool:
    """True quando PEND material waiva gates de sinal (recover soberano)."""
    return _pending_loss_total(metrics) + 1e-12 >= _material_pending_floor(exec_cfg)


def resolve_soft_min_val_accuracy(orch: Any | None = None) -> float:
    """Le soft_min_val_accuracy do deploy_gate SSOT (default 0.53)."""
    if orch is not None:
        cfg = getattr(orch, "config", None)
        if isinstance(cfg, dict):
            dl = cfg.get("deep_learning")
            gate = dl.get("deploy_gate") if isinstance(dl, dict) else None
            if isinstance(gate, dict) and gate.get("soft_min_val_accuracy") is not None:
                return float(gate["soft_min_val_accuracy"])
    return 0.53


def should_skip_acc_floor(
    metrics: dict[str, Any],
    exec_cfg: dict[str, Any] | None,
    *,
    orch: Any | None = None,
    force: bool = False,
) -> bool:
    """True quando ACC live fica abaixo do soft_min e o knob esta ligado."""
    if force or not bool((exec_cfg or {}).get("skip_below_soft_min_acc", False)):
        return False
    raw = metrics.get("val_accuracy")
    if raw is None:
        return False
    try:
        acc = float(raw)
    except (TypeError, ValueError):
        return False
    floor = resolve_soft_min_val_accuracy(orch)
    if acc + 1e-9 >= floor:
        return False
    _mark_skip(
        metrics,
        "acc_floor",
        skip_acc=float(acc),
        skip_acc_floor=float(floor),
    )
    return True


def should_skip_doji(
    metrics: dict[str, Any],
    exec_cfg: dict[str, Any] | None,
    *,
    force: bool = False,
) -> bool:
    """True quando vela M5 fechada stampada nao tem CALL/PUT."""
    if force or not bool((exec_cfg or {}).get("skip_doji", False)):
        return False
    if _pend_waives(metrics, exec_cfg):
        return False
    if not bool(metrics.get("closed_micro_candle_stamped")):
        return False
    if _closed_candle_dir(metrics) is not None:
        return False
    _mark_skip(metrics, "doji")
    return True


def _smart_neg_edge_waive(edge: float, metrics: dict[str, Any]) -> bool:
    """Libera quase-breakeven quando TCN tem lado ativo e loss-clf confirma baixa chance de perda ou bootstrap."""
    if edge < -0.030 or bool(metrics.get("loss_clf_flip")):
        return False
    p_loss = metrics.get("loss_clf_p_loss")
    if p_loss is not None:
        try:
            if float(p_loss) <= 0.485:
                return True
        except (TypeError, ValueError):
            pass
    is_boot = bool(metrics.get("loss_clf_bootstrap")) or metrics.get("loss_clf_flip_blocked") == "bootstrap"
    margin = max(
        float(metrics.get("direction_margin", 0.0) or 0.0),
        float(metrics.get("raw_margin", 0.0) or 0.0),
    )
    return is_boot and margin >= 0.01


def should_skip_neg_edge(
    metrics: dict[str, Any],
    exec_cfg: dict[str, Any] | None,
    *,
    force: bool = False,
) -> bool:
    """True quando Edge calibrado do lado TCN fica <= 0 ou abaixo de min_edge_execute em EXPLORE."""
    if force or not bool((exec_cfg or {}).get("skip_neg_edge", False)):
        return False
    if _pend_waives(metrics, exec_cfg):
        return False
    if bool(metrics.get("loss_clf_flip")) or bool(metrics.get("anti_trend_lock_flip")):
        return False
    raw = metrics.get("cal_side_edge")
    if raw is None:
        raw = metrics.get("edge")
    if raw is None:
        return False
    try:
        edge = float(raw)
    except (TypeError, ValueError):
        return False
    min_edge = 0.0
    if isinstance(exec_cfg, dict):
        try:
            min_edge = max(0.0, float(exec_cfg.get("min_edge_execute", 0.0) or 0.0))
        except (TypeError, ValueError):
            min_edge = 0.0
    floor = max(0.0, min_edge)
    if edge > floor:
        return False
    pend = float(metrics.get("pending_loss_total", 0.0) or 0.0)
    allow_smart = bool((exec_cfg or {}).get("smart_waive_neg_edge", False)) or pend > 0.0
    if allow_smart and _smart_neg_edge_waive(edge, metrics):
        metrics["neg_edge_smart_waived"] = True
        return False
    _mark_skip(metrics, "neg_edge", skip_cal_side_edge=float(edge), min_edge_floor=float(floor))
    return True


def should_skip_exec_vs_candle(
    metrics: dict[str, Any],
    exec_dir: TradeDirection,
    exec_cfg: dict[str, Any] | None,
    *,
    force: bool = False,
) -> bool:
    """True quando EXEC final discordar da vela M5 fechada."""
    if force or not bool((exec_cfg or {}).get("skip_exec_vs_candle", False)):
        return False
    if _pend_waives(metrics, exec_cfg):
        return False
    candle = _closed_candle_dir(metrics)
    if candle is None:
        return False
    exec_name = exec_dir.name
    if candle == exec_name:
        return False
    _mark_skip(
        metrics,
        "exec_vs_candle",
        exec_pre_skip=exec_name,
        candle_dir=candle,
    )
    return True


def should_skip_scale_candle_discord(
    metrics: dict[str, Any],
    exec_cfg: dict[str, Any] | None,
    *,
    force: bool = False,
) -> bool:
    """True quando SCALE resgata TCN≠vela ou candle sobrescreve; waive com PEND."""
    if force or not bool((exec_cfg or {}).get("skip_scale_candle_discord", False)):
        return False
    if _pend_waives(metrics, exec_cfg):
        return False
    candle = _closed_candle_dir(metrics)
    if candle is None:
        return False
    pre = str(metrics.get("exec_direction_pre_scale") or "").strip().upper()
    final = str(metrics.get("exec_direction") or metrics.get("resolved_direction") or "").strip().upper()
    if pre in _VALID and final in _VALID and pre != candle and final == candle and bool(metrics.get("scale_adapted")):
        _mark_skip(
            metrics,
            "scale_rescue",
            exec_pre_scale=pre,
            candle_dir=candle,
            scale_adapt_reason=str(metrics.get("scale_adapt_reason") or ""),
        )
        return True
    reason = str(metrics.get("scale_adapt_reason") or "").strip()
    adapt_from = str(metrics.get("scale_adapt_from") or "").strip().upper()
    adapt_to = str(metrics.get("scale_adapt_to") or "").strip().upper()
    if reason == "candle_vs_tcn" and adapt_from in _VALID and adapt_to in _VALID and adapt_from != adapt_to:
        _mark_skip(
            metrics,
            "scale_candle_conflict",
            scale_adapt_from=adapt_from,
            scale_adapt_to=adapt_to,
            candle_dir=candle,
        )
        return True
    return False


def should_skip_trend_discord(
    metrics: dict[str, Any],
    exec_dir: TradeDirection,
    exec_cfg: dict[str, Any] | None,
    *,
    force: bool = False,
) -> bool:
    """True quando a direcao EXEC discordar simultaneamente de trend_direction e candle M5 em EXPLORE."""
    if force or not bool((exec_cfg or {}).get("skip_trend_discord", False)):
        return False
    if bool(metrics.get("loss_clf_flip")) or bool(metrics.get("anti_trend_lock_flip")):
        return False
    trend = str(metrics.get("trend_direction") or "").strip().upper()
    candle = _closed_candle_dir(metrics)
    if trend not in _VALID or candle not in _VALID:
        return False
    exec_name = exec_dir.name
    if exec_name not in (trend, candle):
        raw_edge = metrics.get("cal_side_edge", metrics.get("edge", 0.0))
        try:
            edge = float(raw_edge or 0.0)
        except (TypeError, ValueError):
            edge = 0.0
        if edge < 0.035:
            _mark_skip(
                metrics,
                "trend_discord",
                exec_pre_skip=exec_name,
                trend_direction=trend,
                candle_dir=candle,
            )
            return True
    return False
