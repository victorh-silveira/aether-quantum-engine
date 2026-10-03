"""Quantificacao nao-parametrica de incerteza preditiva via Conformal Prediction para o TCN."""

from __future__ import annotations

import math
from collections import deque
from typing import Any


DEFAULT_CONFORMAL_ALPHA = 0.10
MIN_CONFORMAL_SAMPLES = 10


def compute_conformal_quantile(scores: list[float] | deque[float], alpha: float = DEFAULT_CONFORMAL_ALPHA) -> float:
    """Calcula o quantil conforme de nao-conformidade com correcao para amostras finitas."""
    n = len(scores)
    if n == 0:
        return 1.0
    val_alpha = max(0.01, min(0.50, float(alpha)))
    sorted_scores = sorted(float(s) for s in scores)
    index = math.ceil((n + 1) * (1.0 - val_alpha)) - 1
    if index >= n:
        return 1.0
    return max(0.0, min(1.0, float(sorted_scores[index])))


class ConformalPredictionBuffer:
    """Buffer deslizante de resíduos de probabilidade para calibracao conforme continua."""

    def __init__(self, maxlen: int = 256):
        self._maxlen = max(32, int(maxlen))
        self._scores: deque[float] = deque(maxlen=self._maxlen)

    def record_outcome(self, predicted_p: float, actual_y: float) -> None:
        """Registra score de erro absoluto |y - p| a partir de um trade liquidado."""
        try:
            p = max(0.0, min(1.0, float(predicted_p)))
            y = 1.0 if bool(actual_y) else 0.0
            score = abs(y - p)
            self._scores.append(score)
        except (TypeError, ValueError):
            pass

    def clear(self) -> None:
        """Limpa historico de scores conformes."""
        self._scores.clear()

    @property
    def sample_count(self) -> int:
        """Quantidade de observacoes acumuladas no buffer."""
        return len(self._scores)

    def evaluate_uncertainty(
        self,
        predicted_p: float,
        *,
        alpha: float = DEFAULT_CONFORMAL_ALPHA,
        min_samples: int = MIN_CONFORMAL_SAMPLES,
    ) -> dict[str, Any]:
        """Avalia intervalo de cobertura e diagnostica incerteza excessiva."""
        p = max(0.0, min(1.0, float(predicted_p)))
        n = len(self._scores)
        if n < max(2, int(min_samples)):
            return {
                "conformal_q": None,
                "p_lower": p,
                "p_upper": p,
                "uncertainty_width": 0.0,
                "is_uncertain": False,
                "sample_size": n,
                "reason": "insufficient_samples",
            }

        q = compute_conformal_quantile(self._scores, alpha=alpha)
        p_lower = max(0.0, p - q)
        p_upper = min(1.0, p + q)
        width = p_upper - p_lower
        is_uncertain = p_lower <= 0.50 <= p_upper

        return {
            "conformal_q": q,
            "p_lower": p_lower,
            "p_upper": p_upper,
            "uncertainty_width": width,
            "is_uncertain": is_uncertain,
            "sample_size": n,
            "reason": "conformal_uncertainty_excessive" if is_uncertain else "ok",
        }


_GLOBAL_CONFORMAL_BUFFER = ConformalPredictionBuffer()


def get_global_conformal_buffer() -> ConformalPredictionBuffer:
    """Retorna singleton de buffer conforme para o ciclo ativo."""
    return _GLOBAL_CONFORMAL_BUFFER


def evaluate_conformal_gate(
    metrics: dict[str, Any] | None,
    predicted_p: float | None,
    *,
    alpha: float = DEFAULT_CONFORMAL_ALPHA,
    min_samples: int = MIN_CONFORMAL_SAMPLES,
    buffer: ConformalPredictionBuffer | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Aplica gate de incerteza conforme decorando metricas do ciclo."""
    buf = buffer if buffer is not None else _GLOBAL_CONFORMAL_BUFFER
    if predicted_p is None:
        return False, {"reason": "missing_prob"}
    evaluation = buf.evaluate_uncertainty(predicted_p, alpha=alpha, min_samples=min_samples)
    if isinstance(metrics, dict):
        metrics["conformal_q"] = evaluation.get("conformal_q")
        metrics["conformal_p_lower"] = evaluation.get("p_lower")
        metrics["conformal_p_upper"] = evaluation.get("p_upper")
        metrics["conformal_uncertainty_width"] = evaluation.get("uncertainty_width")
        metrics["conformal_uncertainty_excessive"] = bool(evaluation.get("is_uncertain"))
    return bool(evaluation.get("is_uncertain")), evaluation
