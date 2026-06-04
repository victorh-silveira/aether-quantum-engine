"""Calibracao de probabilidades do classificador Deep Learning."""

import math
from dataclasses import dataclass


_VAL_ACC_TRUST_FLOOR = 0.50
_TEMPERATURE_MIN = 0.75
_TEMPERATURE_MAX = 2.5


@dataclass
class CalibratorState:
    """Parametros de calibracao Platt e temperatura."""

    temperature: float = 1.0
    platt_a: float = 1.0
    platt_b: float = 0.0


def raw_to_logit(prob: float) -> float:
    """Converte probabilidade em logit estavel numericamente."""
    p = min(max(float(prob), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def logit_to_prob(logit: float) -> float:
    """Converte logit em probabilidade via sigmoide."""
    if logit >= 0:
        z = math.exp(-logit)
        return 1.0 / (1.0 + z)
    z = math.exp(logit)
    return z / (1.0 + z)


def apply_temperature(prob: float, temperature: float) -> float:
    """Suaviza extremos dividindo o logit pela temperatura."""
    temp = max(float(temperature), _TEMPERATURE_MIN)
    return logit_to_prob(raw_to_logit(prob) / temp)


def shrink_toward_fifty(prob: float, val_accuracy: float) -> float:
    """Encolhe conviccao apenas quando val_acc ficar abaixo do piso operacional."""
    val = float(val_accuracy)
    if val <= 0.0:
        val = _VAL_ACC_TRUST_FLOOR
    if val >= _VAL_ACC_TRUST_FLOOR:
        return float(prob)
    gap = _VAL_ACC_TRUST_FLOOR - val
    trust = max(0.72, 1.0 - gap / 0.20)
    return 0.5 + (float(prob) - 0.5) * trust


def apply_platt(prob: float, calibrator: CalibratorState) -> float:
    """Aplica scaling Platt sobre logit da probabilidade."""
    logit = raw_to_logit(prob) * float(calibrator.platt_a) + float(calibrator.platt_b)
    return logit_to_prob(logit)


def apply_calibrator(prob: float, calibrator: CalibratorState) -> float:
    """Combina temperatura e Platt scaling."""
    temp = min(max(float(calibrator.temperature), _TEMPERATURE_MIN), _TEMPERATURE_MAX)
    tempered = apply_temperature(prob, temp)
    return apply_platt(tempered, calibrator)


def raw_side_conviction(raw_prob: float) -> float:
    """Conviccao bruta do lado escolhido (max(p, 1-p))."""
    p = float(raw_prob)
    return max(p, 1.0 - p)


def cap_calibrated_to_raw_band(raw_prob: float, score: float, max_gap: float) -> float:
    """Limita score calibrado para nao exceder raw_side + max_gap."""
    if max_gap <= 0.0:
        return float(score)
    ceiling = min(1.0, raw_side_conviction(raw_prob) + float(max_gap))
    return min(float(score), ceiling)


def calibrate_trade_score(
    raw_prob: float,
    val_accuracy: float,
    calibrator: CalibratorState,
    *,
    max_calibrated_raw_gap: float = 0.25,
    deploy_ok: bool = True,
) -> float:
    """Retorna score calibrado do lado vencedor para gating e stake."""
    calibrated = apply_calibrator(raw_prob, calibrator)
    shrunk = shrink_toward_fifty(calibrated, val_accuracy)
    capped = cap_calibrated_to_raw_band(raw_prob, shrunk, max_calibrated_raw_gap)
    if not deploy_ok:
        return capped
    raw_side = raw_side_conviction(raw_prob)
    floor = raw_side - float(max_calibrated_raw_gap)
    return max(capped, floor)


def calibrate_conviction(raw_prob: float, val_accuracy: float, temperature: float) -> float:
    """Compatibilidade legada com temperatura isolada."""
    calibrator = CalibratorState(temperature=temperature, platt_a=1.0, platt_b=0.0)
    return calibrate_trade_score(raw_prob, val_accuracy, calibrator)


def brier_score(probs: list[float], labels: list[float]) -> float:
    """Erro quadratico medio de probabilidades."""
    if not probs:
        return 1.0
    err = 0.0
    for prob, label in zip(probs, labels, strict=False):
        err += (float(prob) - float(label)) ** 2
    return err / float(len(probs))


def expected_calibration_error(probs: list[float], labels: list[float], *, bins: int = 10) -> float:
    """Estima ECE com histograma de confianca."""
    if not probs:
        return 1.0
    total = float(len(probs))
    ece = 0.0
    for bucket in range(bins):
        lo = bucket / bins
        hi = (bucket + 1) / bins
        idx = [i for i, p in enumerate(probs) if (lo <= float(p) < hi or (hi == 1.0 and float(p) == 1.0))]
        if not idx:
            continue
        avg_conf = sum(float(probs[i]) for i in idx) / len(idx)
        avg_label = sum(float(labels[i]) for i in idx) / len(idx)
        ece += abs(avg_conf - avg_label) * (len(idx) / total)
    return ece


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
    return min(max(best_temp, _TEMPERATURE_MIN), _TEMPERATURE_MAX)


def fit_platt(probs: list[float], labels: list[float], *, steps: int = 40) -> tuple[float, float]:
    """Busca coeficientes Platt minimizando Brier via grid leve."""
    if not probs:
        return 1.0, 0.0
    best_a = 1.0
    best_b = 0.0
    best_err = float("inf")
    for a_step in range(steps):
        a_val = 0.5 + a_step * 0.1
        for b_step in range(steps):
            b_val = -2.0 + b_step * 0.1
            err = 0.0
            for prob, label in zip(probs, labels, strict=False):
                logit = raw_to_logit(prob) * a_val + b_val
                pred = logit_to_prob(logit)
                err += (pred - float(label)) ** 2
            if err < best_err:
                best_err = err
                best_a = a_val
                best_b = b_val
    return best_a, best_b


def fit_calibrator(probs: list[float], labels: list[float]) -> CalibratorState:
    """Ajusta temperatura e Platt no holdout de calibracao."""
    temp = fit_temperature(probs, labels)
    tempered = [apply_temperature(p, temp) for p in probs]
    a_val, b_val = fit_platt(tempered, labels)
    return CalibratorState(temperature=temp, platt_a=a_val, platt_b=b_val)


def calibrator_from_dict(data: dict | None) -> CalibratorState:
    """Reconstrui calibrador a partir de payload de checkpoint."""
    if not data:
        return CalibratorState()
    return CalibratorState(
        temperature=float(data.get("temperature", 1.0)),
        platt_a=float(data.get("platt_a", 1.0)),
        platt_b=float(data.get("platt_b", 0.0)),
    )


def calibrator_to_dict(calibrator: CalibratorState) -> dict:
    """Serializa calibrador para checkpoint."""
    return {
        "temperature": float(calibrator.temperature),
        "platt_a": float(calibrator.platt_a),
        "platt_b": float(calibrator.platt_b),
    }
