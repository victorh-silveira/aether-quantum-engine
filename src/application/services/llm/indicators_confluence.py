"""Confluencia MTF e guarda de distancia EMA para prompts LLM."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

import src.application.services.llm.indicators as ti
from src.application.services.llm import IndicatorConfig
from src.application.services.llm.indicators_numeric import dual_confluence_shrunk_tags


def mtf_confluence_line(
    higher_tf_closes: Sequence[float],
    lower_tf_closes: Sequence[float],
    cfg: IndicatorConfig | None = None,
    *,
    higher_label: str = "M5",
    lower_label: str = "M1",
) -> str:
    """Resume alinhamento Quant entre dois timeframes (Medallion Mandate)."""
    ic = cfg or ti.IndicatorConfig()
    higher = np.asarray(list(higher_tf_closes), dtype=np.float64)
    lower = np.asarray(list(lower_tf_closes), dtype=np.float64)

    h_hurst = ti._hurst_exponent(higher, ic.hurst_window)
    h_zscore = ti._z_score_last(higher, ic.zscore_window)
    l_entropy = ti._shannon_entropy(lower, ic.entropy_bins, ic.entropy_window)
    l_zscore = ti._z_score_last(lower, ic.zscore_window)
    l_hurst = ti._hurst_exponent(lower, ic.hurst_window)

    h_regime = ti._market_regime_quant(h_hurst, h_zscore)
    l_regime = ti._market_regime_quant(l_hurst, l_zscore)

    divergent = (h_regime == "trend_persistente" and l_zscore < -1.0) or (
        h_regime == "mean_reverting" and abs(l_zscore) > 2.0
    )

    if h_regime == "indefinido" or l_regime == "indefinido":
        mtf = "dados_insuficientes"
    elif l_entropy is not None and l_entropy > ic.entropy_threshold:
        mtf = "HIGH_ENTROPY_NOISE (Avoid High Stakes)"
    elif divergent:
        mtf = "DIVERGENCIA_ESTRUTURAL_DETECTADA (Risky)"
    elif h_regime == "trend_persistente" and l_regime == "trend_persistente":
        mtf = "FORTE_CONTINUIDADE_QUANT (High Conviction)"
    elif h_regime == "mean_reverting" or l_regime == "mean_reverting":
        mtf = "ARBITRAGEM_REVERSAO_ESTATISTICA (Counter-Trend)"
    else:
        mtf = "RANDOM_WALK_SEM_EDGE (Noisy)"

    return (
        f"Medallion Confluence: {higher_label}_hurst={h_hurst or 0:.2f} | "
        f"{lower_label}_zscore={l_zscore or 0:.2f} | "
        f"{lower_label}_entropy={l_entropy or 0:.2f} | "
        f"regime={h_regime}/{l_regime} | "
        f"sinal_quant={mtf}"
    )


def dual_confluence_prompt_fragment(line_m30_m5: str, line_m5_m1: str) -> str:
    """Compacta tags de confluencia M30/M5 e M5/M1 para o prompt LLM."""
    return dual_confluence_shrunk_tags(line_m30_m5, line_m5_m1)


def ema_distance_guard_line(
    timeframe_label: str,
    closes: Sequence[float],
    cfg: IndicatorConfig | None = None,
) -> str:
    """Guarda estatistica baseada em Z-Score (substitui distancia EMA)."""
    ic = cfg or ti.IndicatorConfig()
    c = np.asarray(list(closes), dtype=np.float64)
    z = ti._z_score_last(c, ic.zscore_window)
    if z is None:
        return f"{timeframe_label} quant_guard: dados_insuficientes"

    flag = "EXTREMO_ESTATISTICO_ALERTA" if abs(z) > ic.zscore_extreme_threshold else "DENTRO_DA_NORMALIDADE"
    return f"{timeframe_label} quant_guard: zscore={z:.2f} | status={flag}"
