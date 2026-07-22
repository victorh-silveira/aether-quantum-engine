"""Calibracao de probabilidades do classificador Deep Learning."""

import json
import math
from dataclasses import dataclass, field

from aether_paths import repo_path
from src.application.services.deep_learning.dl_calibration_isotonic import apply_isotonic


def _calib_bounds() -> dict[str, float]:
    """Le bounds de calibracao de settings."""
    path = repo_path("config", "settings.json")
    with path.open(encoding="utf-8") as handle:
        full = json.load(handle)
    raw = (full.get("deep_learning") or {}).get("calibration") or {}
    for key in (
        "val_acc_trust_floor",
        "temperature_min",
        "temperature_max",
        "temperature_default",
        "platt_a_default",
        "platt_b_default",
        "trust_blend_floor",
        "trust_blend_span",
    ):
        if key not in raw:
            raise ValueError(f"deep_learning.calibration.{key} obrigatorio")
    return {
        k: float(raw[k])
        for k in (
            "val_acc_trust_floor",
            "temperature_min",
            "temperature_max",
            "temperature_default",
            "platt_a_default",
            "platt_b_default",
            "trust_blend_floor",
            "trust_blend_span",
        )
    }


def temperature_bounds() -> tuple[float, float]:
    """Retorna (temperature_min, temperature_max) de settings."""
    bounds = _calib_bounds()
    return float(bounds["temperature_min"]), float(bounds["temperature_max"])


_METHOD_TEMPERATURE_PLATT = "temperature_platt"
_METHOD_PLATT = "platt"
_METHOD_ISOTONIC = "isotonic"


@dataclass
class CalibratorState:
    """Parametros de calibracao Platt, temperatura ou isotonica."""

    method: str = _METHOD_TEMPERATURE_PLATT
    temperature: float = 1.0
    platt_a: float = 1.0
    platt_b: float = 0.0
    isotonic_x: tuple[float, ...] = field(default_factory=tuple)
    isotonic_y: tuple[float, ...] = field(default_factory=tuple)


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
    temp = max(float(temperature), _calib_bounds()["temperature_min"])
    return logit_to_prob(raw_to_logit(prob) / temp)


def shrink_toward_fifty(prob: float, val_accuracy: float) -> float:
    """Encolhe conviccao apenas quando val_acc ficar abaixo do piso operacional."""
    val = float(val_accuracy)
    if val <= 0.0:
        val = _calib_bounds()["val_acc_trust_floor"]
    if val >= _calib_bounds()["val_acc_trust_floor"]:
        return float(prob)
    gap = _calib_bounds()["val_acc_trust_floor"] - val
    trust = max(float(_calib_bounds()["trust_blend_floor"]), 1.0 - gap / float(_calib_bounds()["trust_blend_span"]))
    return 0.5 + (float(prob) - 0.5) * trust


def apply_platt(prob: float, calibrator: CalibratorState) -> float:
    """Aplica scaling Platt sobre logit da probabilidade."""
    logit = raw_to_logit(prob) * float(calibrator.platt_a) + float(calibrator.platt_b)
    return logit_to_prob(logit)


def apply_calibrator(prob: float, calibrator: CalibratorState) -> float:
    """Aplica calibrador conforme metodo persistido no estado."""
    method = str(calibrator.method or _METHOD_TEMPERATURE_PLATT)
    if method == _METHOD_ISOTONIC:
        return apply_isotonic(prob, calibrator.isotonic_x, calibrator.isotonic_y)
    if method == _METHOD_PLATT:
        return apply_platt(prob, calibrator)
    temp = min(
        max(float(calibrator.temperature), _calib_bounds()["temperature_min"]), _calib_bounds()["temperature_max"]
    )
    tempered = apply_temperature(prob, temp)
    return apply_platt(tempered, calibrator)


def apply_calibrator_stable(prob: float, calibrator: CalibratorState | None) -> float:
    """Aplica calibrador evitando extrapolacao isotonic e flip de lado."""
    raw = float(prob)
    if calibrator is None:
        return raw
    calibrated = float(apply_calibrator(raw, calibrator))
    method = str(calibrator.method or "")
    if method == _METHOD_ISOTONIC and calibrator.isotonic_x:
        xs = calibrator.isotonic_x
        below = raw + 1e-12 < float(xs[0])
        above = raw - 1e-12 > float(xs[-1])
        if below or above:
            return raw
        side_flip = (raw - 0.5) * (calibrated - 0.5) < 0.0
        if side_flip and abs(calibrated - raw) > 0.02:
            return raw
    return calibrated


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
    is_put: bool = False,
) -> float:
    """Retorna score calibrado do lado vencedor para gating e stake."""
    raw_side = 1.0 - raw_prob if is_put else raw_prob
    calibrated = apply_calibrator_stable(raw_side, calibrator)
    shrunk = shrink_toward_fifty(calibrated, val_accuracy)
    capped = cap_calibrated_to_raw_band(raw_prob, shrunk, max_calibrated_raw_gap)
    if not deploy_ok:
        return capped
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


def calibrator_from_dict(data: dict | None) -> CalibratorState:
    """Reconstrui calibrador a partir de payload de checkpoint."""
    if not data:
        return CalibratorState()
    iso_x = data.get("isotonic_x") or ()
    iso_y = data.get("isotonic_y") or ()
    return CalibratorState(
        method=str(data.get("method", _METHOD_TEMPERATURE_PLATT)),
        temperature=float(data.get("temperature", _calib_bounds()["temperature_default"])),
        platt_a=float(data.get("platt_a", _calib_bounds()["platt_a_default"])),
        platt_b=float(data.get("platt_b", _calib_bounds()["platt_b_default"])),
        isotonic_x=tuple(float(v) for v in iso_x),
        isotonic_y=tuple(float(v) for v in iso_y),
    )


def calibrator_to_dict(calibrator: CalibratorState) -> dict:
    """Serializa calibrador para checkpoint."""
    return {
        "method": str(calibrator.method or _METHOD_TEMPERATURE_PLATT),
        "temperature": float(calibrator.temperature),
        "platt_a": float(calibrator.platt_a),
        "platt_b": float(calibrator.platt_b),
        "isotonic_x": [float(v) for v in calibrator.isotonic_x],
        "isotonic_y": [float(v) for v in calibrator.isotonic_y],
    }
