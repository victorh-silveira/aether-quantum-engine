"""Extracao de vetores tabulares para o meta-classificador de stacking."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.application.services.deep_learning.dl_features import FEATURE_DIM
from src.application.services.meta_classifier_cross_symbol import (
    CROSS_SYMBOL_KEYS,
    META_FEATURE_DIM,
    cross_symbol_triplet_from_metrics,
)
from src.application.services.meta_classifier_flow_features import (
    FLOW_FEATURE_KEYS,
    flow_feature_pair_from_metrics,
)


_INDICATOR_KEYS = (
    "hurst",
    "adx",
    "vol_ratio",
    "implied_vol_ratio",
    "bb_width",
    "atr_norm",
    "cmo",
    "keltner",
    "bb_pct_b",
    "rsi",
    "macd",
    "macd_sig",
    "di_diff",
)
_ZSCORE_CLIP = 3.0
_MICRO_MOM_Z_OFFSET = 1
_SHADOW_Z_OFFSET = 3
_MICRO_BLOCK_LEN = 4


def meta_classifier_column_names() -> list[str]:
    """Retorna nomes de colunas para treino LightGBM com features cross-symbol."""
    base = [f"feature_{index}" for index in range(FEATURE_DIM)]
    micro_vol = [
        "micro_bid_ask_spread_momentum",
        "micro_bid_ask_spread_momentum_zscore",
        "volatility_shadow_ratio",
        "volatility_shadow_ratio_zscore",
    ]
    return base + micro_vol + list(CROSS_SYMBOL_KEYS) + list(FLOW_FEATURE_KEYS)


def cross_symbol_conviction_spread(metrics: dict[str, Any]) -> float:
    """Retorna spread de conviccao cruzada anexado em metrics."""
    chunk = metrics.get("cross_symbol_features")
    if isinstance(chunk, dict):
        return float(chunk.get("cross_symbol_prob_delta", 0.0))
    return 0.0


def clip_feature_zscore(value: float) -> float:
    """Aplica saturacao OOD estrita em [-3.0, 3.0] para z-scores do meta 43D."""
    return float(np.clip(float(value), -_ZSCORE_CLIP, _ZSCORE_CLIP))


def _clip_micro_vol_zscores(vector: list[float]) -> list[float]:
    """Blindagem dos indices de z-score de micro-volatilidade no payload 43D."""
    if len(vector) < FEATURE_DIM + _MICRO_BLOCK_LEN:
        return vector
    mom_idx = FEATURE_DIM + _MICRO_MOM_Z_OFFSET
    shadow_idx = FEATURE_DIM + _SHADOW_Z_OFFSET
    vector[mom_idx] = clip_feature_zscore(vector[mom_idx])
    vector[shadow_idx] = clip_feature_zscore(vector[shadow_idx])
    return vector


def _finalize_meta_vector(vector: list[float]) -> list[float]:
    """Garante dimensao rigida META_FEATURE_DIM e clipping dos z-scores micro."""
    if len(vector) > META_FEATURE_DIM:
        vector = vector[:META_FEATURE_DIM]
    while len(vector) < META_FEATURE_DIM:
        vector.append(0.0)
    return _clip_micro_vol_zscores(vector)


def _base_feature_vector(metrics: dict[str, Any]) -> list[float]:
    """Monta vetor base FEATURE_DIM a partir de cache ou indicadores tabulares."""
    stored = metrics.get("feature_vector")
    if isinstance(stored, (list, tuple)) and len(stored) >= FEATURE_DIM:
        v_list = list(stored[:FEATURE_DIM])
        if len(v_list) > 9:
            v_list[8] = clip_feature_zscore(v_list[8])
            v_list[9] = clip_feature_zscore(v_list[9])
        return [float(v) for v in v_list]
    indicators = metrics.get("indicators") if isinstance(metrics.get("indicators"), dict) else {}
    values = []
    for key in _INDICATOR_KEYS:
        val = float(indicators.get(key, 0.0))
        if key in ("bb_width", "atr_norm"):
            val = clip_feature_zscore(val)
        values.append(val)
    raw_prob = metrics.get("calibrated_prob", metrics.get("raw_prob"))
    values.append(float(raw_prob) if raw_prob is not None else 0.5)
    val_accuracy = metrics.get("val_accuracy")
    values.append(float(val_accuracy) if val_accuracy is not None else 0.0)
    edge = metrics.get("edge", metrics.get("calibrated_edge"))
    values.append(float(edge) if edge is not None else 0.0)
    while len(values) < FEATURE_DIM:
        values.append(0.0)
    if len(values) > FEATURE_DIM:
        values = values[:FEATURE_DIM]
    return values


def _micro_vol_block(metrics: dict[str, Any]) -> list[float]:
    """Extrai bloco de micro-volatilidade com z-scores saturados em +-3.0."""
    flow_chunk = metrics.get("flow_features") if isinstance(metrics.get("flow_features"), dict) else {}
    return [
        float(flow_chunk.get("micro_bid_ask_spread_momentum", 0.0)),
        clip_feature_zscore(float(flow_chunk.get("micro_bid_ask_spread_momentum_zscore", 0.0))),
        float(flow_chunk.get("volatility_shadow_ratio", 0.0)),
        clip_feature_zscore(float(flow_chunk.get("volatility_shadow_ratio_zscore", 0.0))),
    ]


def extract_meta_feature_vector(metrics: dict[str, Any]) -> list[float]:
    """Extrai vetor tabular META_FEATURE_DIM=43 espelhado no HTTP :8005."""
    stored_meta = metrics.get("meta_feature_vector")
    if isinstance(stored_meta, (list, tuple)) and len(stored_meta) >= META_FEATURE_DIM:
        vector = _finalize_meta_vector([float(v) for v in stored_meta[:META_FEATURE_DIM]])
        metrics["meta_feature_vector"] = vector
        return vector
    base = _base_feature_vector(metrics)
    cross = cross_symbol_triplet_from_metrics(metrics)
    flow = flow_feature_pair_from_metrics(metrics)
    vector = list(base)
    if META_FEATURE_DIM >= FEATURE_DIM + _MICRO_BLOCK_LEN:
        vector.extend(_micro_vol_block(metrics))
    vector.extend(cross)
    vector.extend(flow)
    vector = _finalize_meta_vector(vector)
    metrics["meta_feature_vector"] = vector
    return vector


def side_payoff_from_probability(probability: float, direction: str) -> float:
    """Converte probabilidade TCN em score lateralizado CALL/PUT."""
    prob = max(0.0, min(1.0, float(probability)))
    if str(direction).upper() == "PUT":
        return max(0.0, min(1.0, 1.0 - prob))
    return prob
