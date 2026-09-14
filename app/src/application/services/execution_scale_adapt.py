"""Adaptacao de lado por SCALE retract/explos/tape (ultima palavra; FLIP sticky)."""

from __future__ import annotations

from typing import Any

from src.domain.models.trade import TradeDirection


_VALID = {TradeDirection.CALL.name, TradeDirection.PUT.name}
_REGIME_RETRACT = "retraction"
_REGIME_EXPLOS = "explosion"
_REASON_RETRACT = "retract_vs_tcn"
_REASON_EXPLOS = "explos_vs_tcn"
_REASON_TAPE = "tape_vs_tcn"
_REASON_FLIP_HOLDS = "flip_holds"
_REASON_EXPLOS_EDGE_FIRM = "explos_edge_firm"


def _side(value: object) -> str | None:
    """Normaliza CALL/PUT ou None."""
    name = str(value or "").strip().upper()
    return name if name in _VALID else None


def _commit_adapt(
    metrics: dict[str, Any],
    *,
    baseline: str,
    target: str,
    reason: str,
) -> TradeDirection:
    """Grava adapt SCALE e devolve o lado alvo (nunca desfaz FLIP)."""
    metrics["scale_adapted"] = True
    metrics["scale_adapt_undid_flip"] = False
    metrics["scale_adapt_reason"] = reason
    metrics["scale_adapt_from"] = baseline
    metrics["scale_adapt_to"] = target
    metrics["exec_direction"] = target
    metrics["resolved_direction"] = target
    return TradeDirection[target]


def _resolve_regime_target(
    metrics: dict[str, Any],
    *,
    require_mili: bool,
) -> tuple[str | None, str | None]:
    """Resolve alvo retract/explos; (target, fail_reason)."""
    target = _side(metrics.get("scale_micro_side")) or _side(metrics.get("scale_mini_bar_dir"))
    if target is None:
        target = _side(metrics.get("scale_mini_dir"))
    if target is None:
        return None, "no_target"
    if require_mili:
        mili = _side(metrics.get("scale_mili_dir"))
        if mili is None or mili != target:
            return None, "mili_mismatch"
    return target, None


def _tcn_cal_edge(metrics: dict[str, Any]) -> float | None:
    """Le Edge calibrado do ciclo para freio de explos."""
    for key in ("cal_side_edge", "neg_edge_tcn_cal_edge", "edge", "cal_edge"):
        raw = metrics.get(key)
        if raw is None:
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return None


def _resolve_tape_target(
    metrics: dict[str, Any],
    baseline: str,
    *,
    require_strong: bool,
) -> tuple[str | None, str | None]:
    """Alvo por fita; (target, fail_reason). Discord so telemetria se require_strong."""
    tape = _side(metrics.get("scale_tape_consensus"))
    if tape is None or tape == baseline:
        return None, None
    strong = bool(metrics.get("scale_tape_strong"))
    if require_strong:
        if not strong:
            return None, "tape_not_strong"
        return tape, None
    discord = bool(metrics.get("scale_discordance"))
    if not (strong or discord):
        return None, None
    return tape, None


def apply_scale_retract_adapt(
    metrics: dict[str, Any],
    tcn_dir: TradeDirection | str,
    *,
    cfg: dict[str, Any] | None = None,
) -> TradeDirection:
    """Fixa EXEC no lado SCALE se retract/explos/tape; nao desfaz FLIP ativo."""
    tcn = _side(getattr(tcn_dir, "name", None) or tcn_dir)
    if tcn is None:
        tcn = _side(metrics.get("tcn_direction"))
    current = _side(metrics.get("exec_direction")) or _side(metrics.get("resolved_direction")) or tcn
    if tcn is None:
        metrics["scale_adapted"] = False
        metrics["scale_adapt_undid_flip"] = False
        metrics.setdefault("scale_adapt_reason", "no_tcn")
        return TradeDirection.CALL if current is None else TradeDirection[current]

    vision = cfg if isinstance(cfg, dict) else {}
    enabled = bool(vision.get("adapt_retract_enabled", False))
    require_mili = bool(vision.get("retraction_require_mili", True))
    require_tape_strong = bool(vision.get("adapt_tape_require_strong", True))
    explos_max_edge = float(vision.get("adapt_explos_max_tcn_edge", 0.05) or 0.05)
    baseline = current or tcn

    metrics["scale_adapted"] = False
    metrics["scale_adapt_undid_flip"] = False
    metrics["scale_adapt_reason"] = "off"
    if not enabled:
        return TradeDirection[baseline]

    regime = str(metrics.get("scale_micro_regime") or "").strip().lower()
    if regime in {_REGIME_RETRACT, _REGIME_EXPLOS}:
        target, fail = _resolve_regime_target(metrics, require_mili=require_mili)
        if fail is not None or target is None:
            metrics["scale_adapt_reason"] = fail or "no_target"
            return TradeDirection[baseline]
        if target == baseline:
            metrics["scale_adapt_reason"] = "aligned"
            return TradeDirection[target]
        if bool(metrics.get("loss_clf_flip")):
            metrics["scale_adapt_reason"] = _REASON_FLIP_HOLDS
            return TradeDirection[baseline]
        if regime == _REGIME_EXPLOS:
            edge = _tcn_cal_edge(metrics)
            if edge is not None and edge > explos_max_edge + 1e-12:
                metrics["scale_adapt_reason"] = _REASON_EXPLOS_EDGE_FIRM
                return TradeDirection[baseline]
        reason = _REASON_EXPLOS if regime == _REGIME_EXPLOS else _REASON_RETRACT
        return _commit_adapt(metrics, baseline=baseline, target=target, reason=reason)

    tape_target, tape_fail = _resolve_tape_target(
        metrics,
        baseline,
        require_strong=require_tape_strong,
    )
    if tape_fail is not None:
        metrics["scale_adapt_reason"] = tape_fail
        return TradeDirection[baseline]
    if tape_target is not None:
        if bool(metrics.get("loss_clf_flip")) and tape_target != baseline:
            metrics["scale_adapt_reason"] = _REASON_FLIP_HOLDS
            return TradeDirection[baseline]
        return _commit_adapt(metrics, baseline=baseline, target=tape_target, reason=_REASON_TAPE)

    metrics["scale_adapt_reason"] = "not_adapt_regime"
    return TradeDirection[baseline]
