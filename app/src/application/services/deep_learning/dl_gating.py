"""Regras de gating para execucao Deep Learning."""

from src.application.services.deep_learning.dl_calibration import raw_side_conviction


_EPS = 1e-6


def unified_execution_score(
    trade_score: float,
    raw_prob: float | None,
    *,
    deploy_ok: bool = True,
    max_calibrated_raw_gap: float = 0.15,
) -> float:
    """Score unico para conviccao e edge com piso alinhado ao raw quando deploy_ok."""
    score = float(trade_score)
    if raw_prob is None or not deploy_ok:
        return score
    raw_side = raw_side_conviction(raw_prob)
    floor = raw_side - float(max_calibrated_raw_gap)
    return max(score, floor)


def resolve_edge(trade_score: float) -> float:
    """Margem do score em relacao a incerteza maxima (0.5)."""
    return abs(float(trade_score) - 0.5)


def calibration_gap(trade_score: float, raw_prob: float) -> float:
    """Diferenca entre score calibrado e conviccao bruta do lado escolhido."""
    return max(0.0, float(trade_score) - raw_side_conviction(raw_prob))


def _raw_side(trade_score: float, raw_prob: float | None) -> float:
    """Obtem conviccao bruta do lado ou usa o score calibrado como fallback."""
    if raw_prob is not None:
        return raw_side_conviction(raw_prob)
    return float(trade_score)


def strong_signal_bypasses_val_acc(
    conviction: float,
    second: float,
    third: float | None = None,
    *,
    bypass_min_conviction: float | None = None,
    bypass_min_edge: float | None = None,
) -> bool:
    """Bypass de val_acc exige conviccao bruta e edge (assinatura legada ou nomeada)."""
    if third is not None and bypass_min_conviction is None:
        val = float(conviction)
        return val + _EPS >= float(second) and resolve_edge(val) + _EPS >= float(third)
    if bypass_min_conviction is None:
        return False
    return float(conviction) + _EPS >= float(bypass_min_conviction) and float(second) + _EPS >= float(
        bypass_min_edge or 0.0
    )


def effective_min_val_accuracy(
    min_val_accuracy: float,
    trade_score: float,
    *,
    bypass_min_conviction: float | None = None,
    bypass_min_edge: float | None = None,
    bypass_min_val_accuracy: float = 0.48,
    allow_bypass: bool = True,
    moderate_min_conviction: float | None = None,
    moderate_min_edge: float | None = None,
    moderate_min_val_accuracy: float | None = None,
    raw_prob: float | None = None,
) -> float:
    """Piso efetivo de val_acc apos bypass forte ou moderado."""
    floor = float(min_val_accuracy)
    if not allow_bypass:
        return floor
    raw_side = _raw_side(trade_score, raw_prob)
    edge = resolve_edge(trade_score)
    strong_ready = bypass_min_conviction is not None and bypass_min_edge is not None
    if strong_ready and strong_signal_bypasses_val_acc(
        raw_side,
        edge,
        bypass_min_conviction=bypass_min_conviction,
        bypass_min_edge=bypass_min_edge,
    ):
        floor = min(floor, float(bypass_min_val_accuracy))
    moderate_ready = (
        moderate_min_conviction is not None and moderate_min_edge is not None and moderate_min_val_accuracy is not None
    )
    if moderate_ready and strong_signal_bypasses_val_acc(
        raw_side,
        edge,
        bypass_min_conviction=moderate_min_conviction,
        bypass_min_edge=moderate_min_edge,
    ):
        floor = min(floor, float(moderate_min_val_accuracy))
    return floor


def should_execute(
    prob: float,
    val_accuracy: float,
    min_conviction: float,
    min_edge: float,
    min_val_accuracy: float,
    **kwargs,
) -> bool:
    """Indica se o candidato pode executar sob os limiares atuais."""
    return gating_block_reason(prob, val_accuracy, min_conviction, min_edge, min_val_accuracy, **kwargs) is None


def _probability_gating_block(
    prob: float,
    val_accuracy: float,
    raw_side: float,
    raw_prob: float | None,
    *,
    saturation_trade_score: float | None = None,
    max_raw_saturation: float | None,
    saturation_min_trade_score: float | None,
    min_raw_conviction: float | None,
    min_conviction_floor: float | None = None,
    deploy_ok: bool = False,
    max_calib_gap: float | None,
    max_val_brier: float | None,
    val_brier: float | None,
    val_acc_high_trust: float,
    brier_untrained_floor: float = 0.99,
    recovery_active: bool = False,
) -> str | None:
    """Avalia bloqueios baseados em probabilidade bruta e metricas de calibracao."""
    sat_floor = float(saturation_min_trade_score if saturation_min_trade_score is not None else 0.58)
    sat_score = float(saturation_trade_score if saturation_trade_score is not None else prob)
    if (
        max_raw_saturation is not None
        and raw_prob is not None
        and raw_side + _EPS > float(max_raw_saturation)
        and sat_score + _EPS >= sat_floor
    ):
        return "saturation"
    if min_raw_conviction is not None and raw_prob is not None and raw_side + _EPS < float(min_raw_conviction):
        conv_floor = float(min_conviction_floor if min_conviction_floor is not None else min_raw_conviction)
        trust_calibrated = deploy_ok and prob + _EPS >= conv_floor
        if not trust_calibrated:
            return "raw_conviction"
    gap_high = (
        max_calib_gap is not None
        and raw_prob is not None
        and calibration_gap(prob, raw_prob) > float(max_calib_gap) + _EPS
    )
    if gap_high and val_accuracy + _EPS < float(val_acc_high_trust):
        return "calib_gap"
    brier_ready = val_brier is not None and (recovery_active or float(val_brier) + _EPS < float(brier_untrained_floor))
    if brier_ready and max_val_brier is not None and float(val_brier) > float(max_val_brier) + _EPS:
        return "brier"
    return None


def _effective_min_conviction(
    min_conviction: float,
    val_accuracy: float,
    raw_side: float,
    *,
    high_val_acc_relax: float = 0.68,
    relaxed_conviction: float = 0.54,
) -> float:
    """Reduz piso de conviccao quando val_acc e raw_side sao fortes."""
    floor = float(min_conviction)
    if val_accuracy + _EPS >= float(high_val_acc_relax) and raw_side + _EPS >= 0.62:
        return min(floor, float(relaxed_conviction))
    return floor


def gating_block_reason(
    prob: float,
    val_accuracy: float,
    min_conviction: float,
    min_edge: float,
    min_val_accuracy: float,
    *,
    raw_prob: float | None = None,
    max_calib_gap: float | None = None,
    min_raw_conviction: float | None = None,
    max_val_brier: float | None = None,
    val_brier: float | None = None,
    max_raw_saturation: float | None = None,
    saturation_min_trade_score: float | None = None,
    high_val_acc_relax: float = 0.68,
    relaxed_conviction: float = 0.54,
    brier_untrained_floor: float = 0.99,
    allow_bypass: bool = True,
    bypass_min_conviction: float | None = None,
    bypass_min_edge: float | None = None,
    bypass_min_val_accuracy: float = 0.48,
    moderate_min_conviction: float | None = None,
    moderate_min_edge: float | None = None,
    moderate_min_val_accuracy: float | None = None,
    val_acc_high_trust: float = 0.65,
    deploy_ok: bool = True,
    recovery_active: bool = False,
    min_conviction_for_raw_bypass: float | None = None,
) -> str | None:
    """Retorna motivo de bloqueio ou None se executavel."""
    gap_lim = float(max_calib_gap if max_calib_gap is not None else 0.15)
    exec_score = unified_execution_score(prob, raw_prob, deploy_ok=deploy_ok, max_calibrated_raw_gap=gap_lim)
    edge = resolve_edge(exec_score)
    raw_side = _raw_side(exec_score, raw_prob)
    early = _probability_gating_block(
        exec_score,
        val_accuracy,
        raw_side,
        raw_prob,
        saturation_trade_score=float(prob),
        max_raw_saturation=max_raw_saturation,
        saturation_min_trade_score=saturation_min_trade_score,
        min_raw_conviction=min_raw_conviction,
        min_conviction_floor=(
            float(min_conviction_for_raw_bypass) if min_conviction_for_raw_bypass is not None else float(min_conviction)
        ),
        deploy_ok=deploy_ok,
        max_calib_gap=max_calib_gap,
        max_val_brier=max_val_brier,
        val_brier=val_brier,
        val_acc_high_trust=val_acc_high_trust,
        brier_untrained_floor=brier_untrained_floor,
        recovery_active=recovery_active,
    )
    if early is not None:
        return early
    min_conv = _effective_min_conviction(
        min_conviction,
        val_accuracy,
        raw_side,
        high_val_acc_relax=high_val_acc_relax,
        relaxed_conviction=relaxed_conviction,
    )
    if exec_score + _EPS < min_conv:
        return "conviction"
    if edge + _EPS < float(min_edge):
        return "edge"
    metrics_untrained = val_brier is not None and float(val_brier) + _EPS >= float(brier_untrained_floor)
    if recovery_active:
        metrics_untrained = False
    floor = effective_min_val_accuracy(
        min_val_accuracy,
        exec_score,
        bypass_min_conviction=bypass_min_conviction,
        bypass_min_edge=bypass_min_edge,
        bypass_min_val_accuracy=bypass_min_val_accuracy,
        allow_bypass=allow_bypass,
        moderate_min_conviction=moderate_min_conviction,
        moderate_min_edge=moderate_min_edge,
        moderate_min_val_accuracy=moderate_min_val_accuracy,
        raw_prob=raw_prob,
    )
    if not metrics_untrained and val_accuracy + _EPS < floor:
        return "val_acc"
    return None


def resolve_gating_thresholds(params: dict, *, recovery_active: bool) -> tuple[float, float, float]:
    """Retorna (min_conviction, min_edge, min_val_accuracy) para modo normal ou recovery."""
    if recovery_active:
        return (
            float(params["recovery_min_conviction"]),
            float(params["recovery_min_edge_margin"]),
            float(params["recovery_min_val_accuracy"]),
        )
    return (
        float(params["min_conviction"]),
        float(params["min_edge_margin"]),
        float(params["min_val_accuracy"]),
    )
