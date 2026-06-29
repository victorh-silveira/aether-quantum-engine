"""Regressao isotonica (PAV) para calibracao de probabilidades DL."""


def fit_isotonic(probs: list[float], labels: list[float]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Ajusta curva monotona via PAV sobre pares (prob, label) ordenados."""
    if not probs:
        return (), ()
    order = sorted(range(len(probs)), key=lambda idx: float(probs[idx]))
    xs = [float(probs[idx]) for idx in order]
    ys = [float(labels[idx]) for idx in order]
    weights = [1.0] * len(xs)
    idx = 0
    while idx < len(ys) - 1:
        if ys[idx] <= ys[idx + 1] + 1e-12:
            idx += 1
            continue
        merged_w = weights[idx] + weights[idx + 1]
        merged_y = (ys[idx] * weights[idx] + ys[idx + 1] * weights[idx + 1]) / merged_w
        xs[idx] = max(xs[idx], xs[idx + 1])
        ys[idx] = merged_y
        weights[idx] = merged_w
        del xs[idx + 1]
        del ys[idx + 1]
        del weights[idx + 1]
        if idx > 0:
            idx -= 1
    return tuple(xs), tuple(ys)


def apply_isotonic(prob: float, xs: tuple[float, ...], ys: tuple[float, ...]) -> float:
    """Interpola probabilidade calibrada na curva isotonica."""
    if not xs or not ys:
        return float(prob)
    p = float(prob)
    result = float(ys[-1])
    if len(xs) == 1 or p < xs[0]:
        result = float(ys[0])
    elif p > xs[-1]:
        result = float(ys[-1])
    else:
        for idx in range(len(xs) - 1):
            lo_x, hi_x = xs[idx], xs[idx + 1]
            if lo_x <= p <= hi_x:
                if abs(hi_x - lo_x) < 1e-12:
                    result = float(ys[idx])
                else:
                    t = (p - lo_x) / (hi_x - lo_x)
                    result = float(ys[idx] + t * (ys[idx + 1] - ys[idx]))
                break
    return max(0.0, min(1.0, result))
