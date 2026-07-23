"""Ajuste de calibradores no holdout de validacao Deep Learning."""

from src.application.services.deep_learning.dl_calibration import (
    _METHOD_ISOTONIC,
    _METHOD_PLATT,
    _METHOD_TEMPERATURE_PLATT,
    CalibratorState,
    apply_calibrator,
    apply_temperature,
    brier_score,
    expected_calibration_error,
    logit_to_prob,
    raw_to_logit,
    temperature_bounds,
)
from src.application.services.deep_learning.dl_calibration_isotonic import fit_isotonic
from src.application.services.deep_learning.dl_sharpness import mean_sharpness, resolve_calibration_sharpness_cfg
from src.domain.math.probability_entropy import binary_entropy, entropy_penalty_factor


def fit_temperature(probs: list[float], labels: list[float]) -> float:
    """Busca temperatura que minimiza Brier na validacao."""
    if not probs:
        return 1.0
    best_temp = 1.0
    best_err = float("inf")
    for step in range(1, 41):
        temp = 0.5 + step * 0.125
        err = 0.0
        for prob, label in zip(probs, labels, strict=False):
            calibrated = apply_temperature(prob, temp)
            err += (calibrated - float(label)) ** 2
        if err < best_err:
            best_err = err
            best_temp = temp
    return min(max(best_temp, temperature_bounds()[0]), temperature_bounds()[1])


def fit_platt_logistic(
    probs: list[float],
    labels: list[float],
    *,
    max_iter: int = 60,
    learning_rate: float = 0.08,
) -> tuple[float, float]:
    """Ajusta coeficientes Platt via regressao logistica 1D sobre logits."""
    if not probs:
        return 1.0, 0.0
    a_val = 1.0
    b_val = 0.0
    n = float(len(probs))
    for _ in range(max_iter):
        grad_a = 0.0
        grad_b = 0.0
        for prob, label in zip(probs, labels, strict=False):
            x_val = raw_to_logit(prob)
            pred = logit_to_prob(a_val * x_val + b_val)
            err = pred - float(label)
            grad_a += err * x_val
            grad_b += err
        a_val -= learning_rate * grad_a / n
        b_val -= learning_rate * grad_b / n
    return a_val, b_val


def fit_platt(probs: list[float], labels: list[float], *, steps: int = 40) -> tuple[float, float]:
    """Compatibilidade: delega para regressao logistica 1D."""
    _ = steps
    return fit_platt_logistic(probs, labels)


def _calibrated_probs(probs: list[float], calibrator: CalibratorState) -> list[float]:
    """Aplica calibrador a lista de probabilidades brutas."""
    return [apply_calibrator(float(p), calibrator) for p in probs]


def _candidate_score(
    calibrator: CalibratorState,
    probs: list[float],
    labels: list[float],
) -> tuple[float, float, float]:
    """Retorna Brier, ECE e sharpness media de um calibrador candidato."""
    calibrated = _calibrated_probs(probs, calibrator)
    return (
        brier_score(calibrated, labels),
        expected_calibration_error(calibrated, labels),
        mean_sharpness(calibrated),
    )


def _build_temperature_platt(probs: list[float], labels: list[float]) -> CalibratorState:
    """Monta calibrador temperatura + Platt."""
    temp = fit_temperature(probs, labels)
    tempered = [apply_temperature(p, temp) for p in probs]
    a_val, b_val = fit_platt_logistic(tempered, labels)
    return CalibratorState(
        method=_METHOD_TEMPERATURE_PLATT,
        temperature=temp,
        platt_a=a_val,
        platt_b=b_val,
    )


def _build_platt(probs: list[float], labels: list[float]) -> CalibratorState:
    """Monta calibrador Platt sobre probabilidades brutas."""
    a_val, b_val = fit_platt_logistic(probs, labels)
    return CalibratorState(method=_METHOD_PLATT, temperature=1.0, platt_a=a_val, platt_b=b_val)


def _build_isotonic(probs: list[float], labels: list[float]) -> CalibratorState:
    """Monta calibrador isotonico sobre probabilidades brutas."""
    xs, ys = fit_isotonic(probs, labels)
    return CalibratorState(method=_METHOD_ISOTONIC, isotonic_x=xs, isotonic_y=ys)


def _select_best_calibrator(
    candidates: list[tuple[CalibratorState, float, float, float]],
    *,
    min_sharpness: float = 0.03,
) -> CalibratorState:
    """Escolhe calibrador por Brier/ECE respeitando piso de sharpness."""
    if not candidates:
        return CalibratorState()
    floor = float(min_sharpness)
    eligible = [item for item in candidates if item[3] + 1e-12 >= floor]
    if eligible:
        ranked = sorted(eligible, key=lambda item: (item[1], item[2]))
        return ranked[0][0]
    ranked = sorted(candidates, key=lambda item: (-item[3], item[1], item[2]))
    return ranked[0][0]


def calibrator_entropy_metrics(
    probs: list[float],
    _labels: list[float],
    calibrator: CalibratorState,
    *,
    calibration_cfg: dict | None = None,
) -> dict[str, float | bool]:
    """Calcula entropia media calibrada e flag de violacao ao teto."""
    cfg = calibration_cfg if isinstance(calibration_cfg, dict) else {}
    ceiling = float(cfg.get("entropy_ceiling", 0.92))
    calibrated = _calibrated_probs(probs, calibrator)
    if not calibrated:
        return {"calibrated_entropy": 0.0, "entropy_violation": False}
    ent_values = [binary_entropy(p) for p in calibrated]
    mean_ent = sum(ent_values) / float(len(ent_values))
    violation = mean_ent > ceiling
    return {"calibrated_entropy": mean_ent, "entropy_violation": violation}


def fit_calibrator(
    probs: list[float],
    labels: list[float],
    *,
    calibration_cfg: dict | None = None,
) -> CalibratorState:
    """Ajusta calibrador no holdout conforme metodo configurado."""
    cfg = calibration_cfg if isinstance(calibration_cfg, dict) else {}
    method = str(cfg.get("method", "auto")).strip().lower()
    isotonic_min = max(3, int(cfg.get("isotonic_min_samples", 20)))
    if not probs:
        return CalibratorState()
    if method == _METHOD_TEMPERATURE_PLATT:
        return _build_temperature_platt(probs, labels)
    if method == _METHOD_PLATT:
        return _build_platt(probs, labels)
    if method == _METHOD_ISOTONIC and len(probs) >= isotonic_min:
        return _build_isotonic(probs, labels)
    sharpness_cfg = resolve_calibration_sharpness_cfg(cfg)
    min_sharpness = float(sharpness_cfg["min_calibration_sharpness"])
    candidates: list[tuple[CalibratorState, float, float, float]] = []
    cal_tp = _build_temperature_platt(probs, labels)
    brier, ece, sharp = _candidate_score(cal_tp, probs, labels)
    candidates.append((cal_tp, brier, ece, sharp))
    cal_platt = _build_platt(probs, labels)
    brier, ece, sharp = _candidate_score(cal_platt, probs, labels)
    candidates.append((cal_platt, brier, ece, sharp))
    if len(probs) >= isotonic_min:
        cal_iso = _build_isotonic(probs, labels)
        brier, ece, sharp = _candidate_score(cal_iso, probs, labels)
        candidates.append((cal_iso, brier, ece, sharp))
    if bool(cfg.get("auto_select_by_brier", True)):
        return _select_best_calibrator(candidates, min_sharpness=min_sharpness)
    return cal_tp


def entropy_weight_penalty(probability: float, *, calibration_cfg: dict | None = None) -> float:
    """Penalizacao [0, 1] para peso DL no resolver."""
    chunk = calibration_cfg if isinstance(calibration_cfg, dict) else {}
    ceiling = float(chunk.get("entropy_ceiling", 0.92))
    floor = float(chunk.get("entropy_floor", 0.0))
    return entropy_penalty_factor(probability, ceiling=ceiling, floor=floor)
