"""Inferencia portavel e verificacao de artefato exclusivo para probabilidade de toque."""

import numpy as np

from src.domain.math.touch_ticks import touch_expected_value
from src.domain.models.touch_policy import TouchPolicy


def predict_touch(bundle: dict, features: list[float]) -> float:
    """Regressao logistica padronizada e calibracao isotonica, sem pickle no runtime."""
    x = np.asarray(features, dtype=float)
    mean, scale, coef = (np.asarray(bundle[key], dtype=float) for key in ("mean", "scale", "coef"))
    if x.shape != (5,) or mean.shape != x.shape or scale.shape != x.shape or coef.shape != x.shape:
        raise ValueError("Schema Touch incompativel")
    if not np.isfinite([x, mean, scale, coef]).all() or np.any(scale <= 0):
        raise ValueError("Numeros invalidos no modelo Touch")
    z = float(np.dot((x - mean) / scale, coef) + float(bundle["intercept"]))
    raw = 1 / (1 + np.exp(-np.clip(z, -40, 40)))
    prob = float(np.interp(raw, bundle["cal_x"], bundle["cal_y"]))
    if not np.isfinite(prob) or not 0 <= prob <= 1:
        raise ValueError("Calibracao Touch invalida")
    return prob


def validate_touch_bundle(bundle: dict, policy: TouchPolicy, now_ms: float) -> None:
    """Aprovacao exige avaliacao economica OOS com propostas historicas e ticks."""
    report = bundle.get("oos", {})
    if (
        bundle.get("schema") != "touch_ticks_v1"
        or bundle.get("policy_hash") != policy.fingerprint()
        or bundle.get("qualified") is not True
        or report.get("source") != "quoted_tick_replay"
        or int(report.get("trades", 0)) < policy.min_oos_trades
        or not np.isfinite(float(report.get("return_lcb90", float("nan"))))
        or float(report.get("return_lcb90", 0)) <= 0
        or not 0 <= now_ms - float(bundle.get("trained_ms", 0)) <= policy.model_max_age_days * 86400000
    ):
        raise ValueError("Modelo Touch ausente, vencido, incompativel ou sem qualificacao OOS")


def select_touch_quote(bundle: dict, quotes: list[dict], policy: TouchPolicy) -> tuple[dict, float, float] | None:
    """Seleciona EV positivo entre ambas as familias/barreiras sem ajustar p para gerar trades."""
    candidates = []
    for quote in quotes:
        probability = predict_touch(bundle, quote["features"])
        edge = touch_expected_value(probability, quote)
        if edge > policy.min_edge:
            candidates.append((quote, probability, edge))
    return max(candidates, key=lambda row: row[2]) if candidates else None
