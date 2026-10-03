"""Monitoramento de drift de dados e conceito via Population Stability Index e momentos estocasticos."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class DriftLevel(StrEnum):
    """Niveis de classificacao de drift estatistico."""

    NO_DRIFT = "NO_DRIFT"
    MODERATE_DRIFT = "MODERATE_DRIFT"
    SEVERE_DRIFT = "SEVERE_DRIFT"


@dataclass(frozen=True)
class DriftReport:
    """Relatorio analitico contendo diagnostico de estabilidade temporal."""

    psi: float
    level: DriftLevel
    kurtosis_delta: float
    skewness_delta: float
    is_actionable: bool
    recommendation: str


def _calculate_moments(data: np.ndarray) -> tuple[float, float]:
    """Calcula skewness e excess kurtosis a partir de momentos centrais amostrais."""
    if len(data) < 4:
        return 0.0, 0.0
    mean = float(np.mean(data))
    diff = data - mean
    var = float(np.mean(diff**2))
    if var < 1e-12:
        return 0.0, 0.0
    std = np.sqrt(var)
    skew = float(np.mean((diff / std) ** 3))
    kurt = float(np.mean((diff / std) ** 4) - 3.0)
    return skew, kurt


class DataDriftMonitor:
    """Monitor de estabilidade de distribuicao para series temporais de mercado."""

    def __init__(
        self,
        reference_returns: np.ndarray,
        *,
        num_bins: int = 10,
        psi_moderate_threshold: float = 0.10,
        psi_severe_threshold: float = 0.25,
    ) -> None:
        """Inicializa monitor com bins baseados nos quantis da serie de referencia."""
        ref = np.asarray(reference_returns, dtype=np.float64)
        valid_ref = ref[np.isfinite(ref)]
        if len(valid_ref) < num_bins * 2:
            self._is_ready = False
            self._bin_edges = np.empty(0, dtype=np.float64)
            self._ref_pct = np.empty(0, dtype=np.float64)
            self._ref_skew = 0.0
            self._ref_kurt = 0.0
        else:
            self._is_ready = True
            quantiles = np.linspace(0.0, 1.0, num_bins + 1)
            raw_edges = np.quantile(valid_ref, quantiles)
            raw_edges[0] = -np.inf
            raw_edges[-1] = np.inf
            self._bin_edges = raw_edges
            counts, _ = np.histogram(valid_ref, bins=self._bin_edges)
            self._ref_pct = np.maximum(counts / len(valid_ref), 1e-4)
            self._ref_skew, self._ref_kurt = _calculate_moments(valid_ref)

        self._num_bins = num_bins
        self._mod_thresh = psi_moderate_threshold
        self._sev_thresh = psi_severe_threshold

    @property
    def is_ready(self) -> bool:
        """Indica se a serie de referencia possui volume estatistico suficiente."""
        return self._is_ready

    def evaluate(self, recent_returns: np.ndarray) -> DriftReport:
        """Avalia a divergencia de distribuicao da janela recente contra a referencia."""
        rec = np.asarray(recent_returns, dtype=np.float64)
        valid_rec = rec[np.isfinite(rec)]

        if not self._is_ready or len(valid_rec) < self._num_bins:
            return DriftReport(
                psi=0.0,
                level=DriftLevel.NO_DRIFT,
                kurtosis_delta=0.0,
                skewness_delta=0.0,
                is_actionable=False,
                recommendation="insufficient_data",
            )

        counts, _ = np.histogram(valid_rec, bins=self._bin_edges)
        actual_pct = np.maximum(counts / len(valid_rec), 1e-4)

        psi = float(np.sum((actual_pct - self._ref_pct) * np.log(actual_pct / self._ref_pct)))
        rec_skew, rec_kurt = _calculate_moments(valid_rec)
        skew_delta = rec_skew - self._ref_skew
        kurt_delta = rec_kurt - self._ref_kurt

        if psi > self._sev_thresh:
            level = DriftLevel.SEVERE_DRIFT
            actionable = True
            rec_text = "pause_trading_severe_distribution_shift"
        elif psi > self._mod_thresh:
            level = DriftLevel.MODERATE_DRIFT
            actionable = True
            rec_text = "reduce_sizing_moderate_regime_shift"
        else:
            level = DriftLevel.NO_DRIFT
            actionable = False
            rec_text = "maintain_operational_regime"

        return DriftReport(
            psi=max(0.0, psi),
            level=level,
            kurtosis_delta=kurt_delta,
            skewness_delta=skew_delta,
            is_actionable=actionable,
            recommendation=rec_text,
        )
