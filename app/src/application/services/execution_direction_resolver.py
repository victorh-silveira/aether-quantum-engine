"""Motor unificado de direcao CALL/PUT por scoring de indicadores e DL."""

from __future__ import annotations

from functools import partial

from src.application.services.execution_direction_cross_corr import adjust_dl_weight_with_correlation
from src.application.services.execution_direction_expansion_veto import apply_expansion_inversion_veto
from src.application.services.execution_direction_resolver_bias import (
    exhaustion_bias as _exhaustion_bias,
    indicator_regime_bias as _indicator_regime_bias,
    trend_bias as _trend_bias,
    val_accuracy_bias as _val_accuracy_bias,
)
from src.application.services.execution_direction_scoring import (
    accumulate_direction_scores,
    finalize_direction_metrics,
)
from src.application.services.execution_direction_softmax_conflict import resolve_softmax_exhaustion_conflict
from src.application.services.execution_entropy_adaptive import resolve_dl_entropy_penalty
from src.application.services.execution_exhaustion_conflict import exhaustion_conflict_penalty
from src.application.services.execution_exhaustion_hard_gate import (
    dl_weight_retention,
    hard_gate_score_penalty,
)
from src.domain.models.trade import TradeDirection


def _direction_prob(entry: dict) -> float | None:
    """Retorna probabilidade calibrada de CALL quando disponivel."""
    metrics = entry.get("metrics") or {}
    calibrated = metrics.get("calibrated_prob")
    if calibrated is not None:
        return float(calibrated)
    raw = metrics.get("raw_prob")
    if raw is None:
        return None
    return float(raw)


def _direction_pivot(metrics: dict) -> float:
    """Pivot CALL/PUT: medio dos thresholds dinamicos ou 0.5."""
    call_th = metrics.get("dynamic_call_threshold")
    put_th = metrics.get("dynamic_put_threshold")
    if call_th is not None and put_th is not None:
        return (float(call_th) + float(put_th)) * 0.5
    return 0.5


def infer_dl_direction(entry: dict) -> TradeDirection | None:
    """Obtem direcao prevista pelo DL ou infere a partir da probabilidade calibrada."""
    direction = entry.get("direction")
    if direction is not None:
        return direction
    metrics = entry.get("metrics") or {}
    prob = _direction_prob(entry)
    if prob is None:
        return None
    pivot = _direction_pivot(metrics)
    return TradeDirection.CALL if float(prob) > pivot else TradeDirection.PUT


_TECHNICAL_BLOCKS = frozenset({"data", "predict_error", "training"})
_DEFAULT_WEIGHTS = {
    "dl_raw_weight": 0.45,
    "val_accuracy_weight": 0.18,
    "trend_weight": 0.15,
    "exhaustion_weight": 0.12,
    "indicator_regime_weight": 0.10,
}


def is_technically_blocked(entry: dict) -> bool:
    """Indica bloqueio absoluto por falha tecnica."""
    metrics = entry.get("metrics") or {}
    if metrics.get("deploy_ok") is False:
        return True
    gate = str(metrics.get("gate_reason") or "")
    return gate in _TECHNICAL_BLOCKS


def _scoring_weights(cfg: dict) -> dict[str, float]:
    """Mescla pesos de direction_scoring com defaults do resolver."""
    merged = dict(_DEFAULT_WEIGHTS)
    chunk = cfg.get("direction_scoring") if isinstance(cfg, dict) else {}
    if isinstance(chunk, dict):
        for key in _DEFAULT_WEIGHTS:
            if key in chunk:
                merged[key] = float(chunk[key])
    return merged


def _clamp01(value: float) -> float:
    """Limita valor ao intervalo [0, 1]."""
    return max(0.0, min(1.0, float(value)))


def _dl_call_put_scores(
    entry: dict,
    weights: dict,
    *,
    calibration_cfg: dict | None = None,
    dynamic_cfg: dict | None = None,
    exec_cfg: dict | None = None,
) -> tuple[float, float]:
    """Calcula contribuicao lateralizada da probabilidade calibrada no score CALL/PUT."""
    metrics = entry.get("metrics") or {}
    prob = _direction_prob(entry)
    dl_dir = infer_dl_direction(entry)
    if prob is None:
        if dl_dir is None:
            return 0.5, 0.5
        prob = 0.55 if dl_dir == TradeDirection.CALL else 0.45
    prob_f = _clamp01(float(prob))
    pivot = _direction_pivot(metrics)
    w = float(weights["dl_raw_weight"])
    penalty, ent, ceiling_eff = resolve_dl_entropy_penalty(
        prob_f,
        metrics,
        calibration_cfg=calibration_cfg,
        dynamic_cfg=dynamic_cfg,
    )
    if metrics.get("entropy_violation"):
        penalty = min(1.0, max(penalty, 0.5))
    w_eff = w * (1.0 - penalty * penalty)
    if dl_dir is not None:
        retention = dl_weight_retention(metrics, dl_dir, cfg=exec_cfg)
        w_eff *= retention
        metrics["exhaustion_dl_retention"] = retention
        metrics["exhaustion_hard_gate"] = retention < 1.0
    metrics["dl_weight_penalty"] = penalty
    metrics["direction_entropy"] = ent
    metrics["entropy_ceiling_effective"] = ceiling_eff
    return 0.5 + (prob_f - pivot) * w_eff, 0.5 + (pivot - prob_f) * w_eff


def _low_val_accuracy_bias(entry: dict, metrics: dict, weights: dict) -> tuple[float, float, str | None]:
    """Inverte bias lateral quando val_accuracy esta abaixo de 0.5."""
    val_acc = float(metrics.get("val_accuracy", 0.5))
    if val_acc >= 0.50:
        return 0.5, 0.5, None
    dl_dir = infer_dl_direction(entry)
    if dl_dir is None:
        return 0.5, 0.5, None
    w = float(weights["val_accuracy_weight"]) * 0.5
    inverted = TradeDirection.PUT if dl_dir == TradeDirection.CALL else TradeDirection.CALL
    if inverted == TradeDirection.CALL:
        return 0.5 + w, 0.5 - w, "low_val_flip"
    return 0.5 - w, 0.5 + w, "low_val_flip"


def resolve_execution_direction(
    entry: dict,
    *,
    exec_cfg: dict | None = None,
    calibration_cfg: dict | None = None,
    recovery_active: bool = False,
    symbol: str | None = None,
    corr_matrix: dict[tuple[str, str], float] | None = None,
) -> tuple[TradeDirection, dict] | None:
    """Resolve CALL ou PUT por score composto; retorna None apenas se sem probabilidade."""
    if is_technically_blocked(entry):
        return None
    dl_dir = infer_dl_direction(entry)
    if dl_dir is None:
        return None
    metrics = dict(entry.get("metrics") or {})
    entry = {**entry, "metrics": metrics}
    cfg = exec_cfg if isinstance(exec_cfg, dict) else {}
    cal_cfg = calibration_cfg if isinstance(calibration_cfg, dict) else {}
    dynamic_cfg = cfg.get("dynamic_threshold") if isinstance(cfg.get("dynamic_threshold"), dict) else {}
    weights = _scoring_weights(cfg)
    qchunk = cfg.get("quality_gate") if isinstance(cfg.get("quality_gate"), dict) else {}
    min_margin = float(qchunk.get("min_direction_margin", 0.05)) if isinstance(qchunk, dict) else 0.05
    if corr_matrix and symbol:
        weights = adjust_dl_weight_with_correlation(
            weights,
            str(symbol),
            metrics,
            corr_matrix,
            min_margin=min_margin,
        )
    dl_scores_fn = partial(
        _dl_call_put_scores,
        calibration_cfg=cal_cfg,
        dynamic_cfg=dynamic_cfg,
        exec_cfg=cfg,
    )
    exhaustion_bias_fn = partial(_exhaustion_bias, cfg=cfg)
    call_score, put_score, hints = accumulate_direction_scores(
        entry,
        metrics,
        weights,
        recovery_active=recovery_active,
        bias_fns=(_trend_bias, exhaustion_bias_fn, _indicator_regime_bias),
        dl_scores_fn=dl_scores_fn,
        val_bias_fn=_val_accuracy_bias,
        low_val_fn=_low_val_accuracy_bias,
    )
    exec_dir = TradeDirection.CALL if call_score + 1e-9 >= put_score else TradeDirection.PUT
    finalize_direction_metrics(
        metrics,
        call_score=call_score,
        put_score=put_score,
        hints=hints,
        dl_dir=dl_dir,
        exec_dir=exec_dir,
        clamp01=_clamp01,
    )
    exec_dir, hints = apply_expansion_inversion_veto(
        exec_dir,
        dl_dir,
        hints,
        metrics,
        exec_cfg=cfg,
        clamp01=_clamp01,
    )
    conflict, penalty = exhaustion_conflict_penalty(metrics, dl_dir, cfg=cfg)
    if metrics.get("exhaustion_hard_gate"):
        penalty = max(penalty, hard_gate_score_penalty(cfg=cfg))
        conflict = True
        if "exhaustion_hard_gate" not in hints:
            hints.append("exhaustion_hard_gate")
    metrics["exhaustion_conflict"] = conflict
    metrics["exhaustion_penalty"] = penalty
    metrics["direction_hints"] = hints
    exec_dir, metrics = resolve_softmax_exhaustion_conflict(
        exec_dir,
        dl_dir,
        metrics,
        call_score=call_score,
        put_score=put_score,
        exec_cfg=cfg,
    )
    return exec_dir, metrics
