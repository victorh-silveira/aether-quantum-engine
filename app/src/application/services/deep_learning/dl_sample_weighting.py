"""Pesos de treino: balanceamento de classe e recencia temporal."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.domain.config_knobs import merge_settings_block, require_bool, require_float, require_int, require_keys


_SAMPLE_WEIGHTING_KEYS = (
    "class_balance_enabled",
    "class_balance_eps",
    "recency_enabled",
    "recency_half_life_n",
)


def parse_sample_weighting_config(dl_config: dict | None = None) -> dict[str, Any]:
    """Resolve sample_weighting com merge fail-closed sobre o SSOT."""
    override = dl_config.get("sample_weighting") if isinstance(dl_config, dict) else None
    raw = merge_settings_block(
        ("deep_learning", "sample_weighting"),
        override if isinstance(override, dict) else None,
    )
    block = require_keys(raw, _SAMPLE_WEIGHTING_KEYS, "deep_learning.sample_weighting")
    return {
        "class_balance_enabled": require_bool(block, "class_balance_enabled"),
        "class_balance_eps": max(0.0, require_float(block, "class_balance_eps")),
        "recency_enabled": require_bool(block, "recency_enabled"),
        "recency_half_life_n": max(1, require_int(block, "recency_half_life_n")),
    }


def label_call_fraction(y: np.ndarray | list[float]) -> float:
    """Fracao de labels CALL (y>=0.5) na amostra."""
    arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.5
    return float(np.mean(arr >= 0.5))


def apply_class_balance_weights(
    weights: list[float],
    y_train: np.ndarray | list[float],
    *,
    enabled: bool = True,
    imbalance_eps: float = 0.05,
) -> list[float]:
    """Repondera CALL/PUT pela taxa inversa quando o desequilibrio supera eps."""
    if not enabled or len(weights) == 0:
        return list(weights)
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    if y.size != len(weights):
        return list(weights)
    pos_rate = label_call_fraction(y)
    if abs(pos_rate - 0.5) <= float(imbalance_eps):
        return list(weights)
    pos_w = max(1.0 - pos_rate, 1e-6) / max(pos_rate, 1e-6)
    neg_w = max(pos_rate, 1e-6) / max(1.0 - pos_rate, 1e-6)
    return [float(weights[i]) * (pos_w if float(y[i]) >= 0.5 else neg_w) for i in range(len(weights))]


def apply_recency_half_life(
    weights: list[float],
    *,
    half_life_n: int,
    enabled: bool = True,
) -> list[float]:
    """Aplica decaimento exponencial: amostras recentes (fim da lista) pesam mais."""
    n = len(weights)
    if not enabled or n == 0:
        return list(weights)
    hl = max(1, int(half_life_n))
    out: list[float] = []
    for i in range(n):
        age = float(n - 1 - i)
        decay = float(0.5 ** (age / float(hl)))
        out.append(float(weights[i]) * max(decay, 1e-6))
    return out


def align_sample_weights(
    sample_weights: list[float] | None,
    *,
    full_n: int,
    train_index: np.ndarray | slice | list[int],
) -> list[float]:
    """Alinha pesos ao split de treino (full_n ou ja no tamanho do train)."""
    train_idx = np.asarray(list(range(full_n))[train_index], dtype=np.int64)
    train_n = int(train_idx.size)
    if train_n <= 0:
        return []
    if sample_weights is None:
        return [1.0] * train_n
    if len(sample_weights) == train_n:
        return [float(w) for w in sample_weights]
    if len(sample_weights) == full_n:
        return [float(sample_weights[int(i)]) for i in train_idx]
    return [1.0] * train_n


def compose_train_weights(
    sample_weights: list[float] | None,
    y_train: np.ndarray | list[float],
    *,
    full_n: int,
    train_index: np.ndarray | slice | list[int],
    weighting_cfg: dict[str, Any] | None = None,
) -> list[float]:
    """Compõe pesos de outcome + balanceamento de classe + half-life de recencia."""
    cfg = weighting_cfg if isinstance(weighting_cfg, dict) else {}
    weights = align_sample_weights(sample_weights, full_n=full_n, train_index=train_index)
    weights = apply_class_balance_weights(
        weights,
        y_train,
        enabled=bool(cfg.get("class_balance_enabled", True)),
        imbalance_eps=float(cfg.get("class_balance_eps", 0.05)),
    )
    return apply_recency_half_life(
        weights,
        half_life_n=int(cfg.get("recency_half_life_n", 2000)),
        enabled=bool(cfg.get("recency_enabled", True)),
    )


def minority_class_recall(
    y_true: np.ndarray | list[float],
    y_pred_call: np.ndarray | list[bool],
) -> float:
    """Recall da classe minoritaria nos labels (CALL ou PUT)."""
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred_call, dtype=bool).reshape(-1)
    if y.size == 0 or pred.size != y.size:
        return 1.0
    call_mask = y >= 0.5
    put_mask = ~call_mask
    n_call = int(call_mask.sum())
    n_put = int(put_mask.sum())
    if n_call == 0 or n_put == 0:
        return 0.0
    recall_call = float((pred[call_mask]).mean()) if n_call else 0.0
    recall_put = float((~pred[put_mask]).mean()) if n_put else 0.0
    if n_call <= n_put:
        return recall_call
    return recall_put
