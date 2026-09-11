"""Adaptacao de lado por SCALE retract/explos/tape (ultima palavra apos FLIP; sem SKIP)."""

from __future__ import annotations

from typing import Any

from src.domain.models.trade import TradeDirection


_VALID = {TradeDirection.CALL.name, TradeDirection.PUT.name}
_REGIME_RETRACT = "retraction"
_REGIME_EXPLOS = "explosion"
_REASON_RETRACT = "retract_vs_tcn"
_REASON_RETRACT_HOLDS = "retract_holds"
_REASON_EXPLOS = "explos_vs_tcn"
_REASON_EXPLOS_HOLDS = "explos_holds"
_REASON_TAPE = "tape_vs_tcn"
_REASON_TAPE_HOLDS = "tape_holds"


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
    """Grava adapt SCALE e devolve o lado alvo."""
    undid_flip = bool(metrics.get("loss_clf_flip")) and target != baseline
    metrics["scale_adapted"] = True
    metrics["scale_adapt_undid_flip"] = undid_flip
    metrics["scale_adapt_reason"] = reason
    metrics["scale_adapt_from"] = baseline
    metrics["scale_adapt_to"] = target
    metrics["exec_direction"] = target
    metrics["resolved_direction"] = target
    return TradeDirection[target]


def _reason_regime(regime: str, *, undid_flip: bool) -> str:
    """Token why= para retract/explos."""
    if regime == _REGIME_EXPLOS:
        return _REASON_EXPLOS_HOLDS if undid_flip else _REASON_EXPLOS
    return _REASON_RETRACT_HOLDS if undid_flip else _REASON_RETRACT


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
    """Fixa EXEC no lado SCALE se retract/explos/tape confirmado; pode desfazer FLIP."""
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
        undid = bool(metrics.get("loss_clf_flip")) and target != baseline
        return _commit_adapt(
            metrics,
            baseline=baseline,
            target=target,
            reason=_reason_regime(regime, undid_flip=undid),
        )

    tape_target, tape_fail = _resolve_tape_target(
        metrics,
        baseline,
        require_strong=require_tape_strong,
    )
    if tape_fail is not None:
        metrics["scale_adapt_reason"] = tape_fail
        return TradeDirection[baseline]
    if tape_target is not None:
        undid = bool(metrics.get("loss_clf_flip")) and tape_target != baseline
        return _commit_adapt(
            metrics,
            baseline=baseline,
            target=tape_target,
            reason=_REASON_TAPE_HOLDS if undid else _REASON_TAPE,
        )

    metrics["scale_adapt_reason"] = "not_adapt_regime"
    return TradeDirection[baseline]
