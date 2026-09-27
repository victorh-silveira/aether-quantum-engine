"""Features causais e labels de primeiro toque sobre trajetorias completas."""

import numpy as np

from src.domain.models.touch_policy import TouchPolicy


def tick_arrays(ticks) -> tuple[np.ndarray, np.ndarray]:
    """Rejeita precos invalidos, duplicatas e inversoes de tempo; nao interpola."""
    data = np.asarray(ticks, dtype=float)
    if data.ndim != 2 or data.shape[1] != 2 or len(data) < 2:
        raise ValueError("Ticks insuficientes")
    if not np.isfinite(data).all() or np.any(data <= 0) or np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError("Ticks invalidos ou fora de ordem")
    return data[:, 0], data[:, 1]


def touch_features(ticks, barrier: float, policy: TouchPolicy) -> list[float]:
    """Distancia, drift, volatilidade e intensidade usam exclusivamente ticks passados."""
    epochs, prices = tick_arrays(ticks)
    start = int(np.searchsorted(epochs, epochs[-1] - policy.history_seconds * 1000))
    epochs, prices = epochs[start:], prices[start:]
    if epochs[-1] - epochs[0] < policy.history_seconds * 1000 - 1000:
        raise ValueError("Warmup Touch incompleto")
    if np.max(np.diff(epochs)) > policy.max_tick_gap_ms:
        raise ValueError("Lacuna no historico Touch")
    diffs = np.diff(prices)
    sigma = float(np.std(diffs))
    if sigma <= 1e-12 or not np.isfinite(barrier) or barrier <= 0:
        raise ValueError("Volatilidade/barreira invalida")
    scale = sigma * np.sqrt(policy.duration_seconds)
    return [
        float((barrier - prices[-1]) / scale),
        float(np.mean(diffs) * policy.duration_seconds / scale),
        float(np.log(sigma / prices[-1])),
        float(np.std(diffs[-30:]) / sigma),
        float(len(diffs) * 1000 / (epochs[-1] - epochs[0])),
    ]


def candidate_barriers(ticks, policy: TouchPolicy) -> list[float]:
    """Barreiras absolutas compartilhadas por Touch e No Touch, sem inversao do lado."""
    _, prices = tick_arrays(ticks)
    sigma = float(np.std(np.diff(prices[-policy.history_seconds :])))
    barriers = [
        round(float(prices[-1] + side * k * sigma * np.sqrt(policy.duration_seconds)), 2)
        for k in policy.barrier_sigma
        for side in (-1, 1)
    ]
    return [b for b in dict.fromkeys(barriers) if b > 0 and b != prices[-1]]


def replay_touch_labels(ticks, quote: dict, policy: TouchPolicy) -> list[int]:
    """Reproduz toque inclusivo em cenarios explicitos de latencia, sem chamar proxy de broker."""
    epochs, prices = tick_arrays(ticks)
    barrier = float(quote["barrier"])
    upper = float(quote["features"][0]) > 0
    labels = []
    for latency in policy.latency_ms:
        start = float(quote["decision_ms"]) + latency
        end = start + policy.duration_seconds * 1000
        left, right = int(np.searchsorted(epochs, start)), int(np.searchsorted(epochs, end, side="right"))
        if left == 0 or right >= len(epochs) or right <= left:
            raise ValueError("Janela de settlement incompleta")
        if np.max(np.diff(epochs[left - 1 : right + 1])) > policy.max_tick_gap_ms:
            raise ValueError("Lacuna no settlement Touch")
        path = prices[left:right]
        labels.append(int(np.any(path >= barrier) if upper else np.any(path <= barrier)))
    return labels


def touch_expected_value(probability: float, quote: dict) -> float:
    """Retorno por unidade apostada usando payout bruto da propria proposta."""
    ask, payout = float(quote["ask_price"]), float(quote["payout"])
    if not np.isfinite([probability, ask, payout]).all() or not 0 <= probability <= 1 or not 0 < ask < payout:
        raise ValueError("Probabilidade ou cotacao invalida")
    p_win = probability if quote["contract_type"] == "ONETOUCH" else 1 - probability
    if quote["contract_type"] not in {"ONETOUCH", "NOTOUCH"}:
        raise ValueError("Familia de contrato invalida")
    return float(p_win * payout / ask - 1)
