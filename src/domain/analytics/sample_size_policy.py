"""Politica operacional da Lei dos Grandes Numeros vs vies dos Pequenos Numeros."""

from __future__ import annotations

from math import sqrt
from typing import Any

from src.domain.config_knobs import merge_settings_block, require_bool, require_float, require_int, require_keys


_KEYS = (
    "enabled",
    "small_n_max",
    "large_n_min",
    "evidence_n_min",
    "reliability_half_life_n",
    "z_sig_threshold",
    "explore_stake_scale_floor",
    "calib_soft_min_n",
    "toxic_side_n_min",
    "prior_wr",
)

_CACHE: dict[str, Any] = {"policy": None}


def reset_sample_size_policy_cache() -> None:
    """Limpa o cache SSOT de sample_size_policy."""
    _CACHE["policy"] = None


def load_sample_size_policy(override: dict[str, Any] | None = None) -> dict[str, Any]:
    """Carrega e valida orchestrator.execution.sample_size_policy do SSOT."""
    cached = _CACHE.get("policy")
    if override is None and cached is not None:
        return dict(cached)
    raw = require_keys(
        merge_settings_block(("orchestrator", "execution", "sample_size_policy"), override),
        _KEYS,
        "orchestrator.execution.sample_size_policy",
    )
    resolved = {
        "enabled": require_bool(raw, "enabled"),
        "small_n_max": max(2, require_int(raw, "small_n_max")),
        "large_n_min": max(10, require_int(raw, "large_n_min")),
        "evidence_n_min": max(4, require_int(raw, "evidence_n_min")),
        "reliability_half_life_n": max(1, require_int(raw, "reliability_half_life_n")),
        "z_sig_threshold": max(0.0, require_float(raw, "z_sig_threshold")),
        "explore_stake_scale_floor": max(0.05, min(1.0, require_float(raw, "explore_stake_scale_floor"))),
        "calib_soft_min_n": max(1, require_int(raw, "calib_soft_min_n")),
        "toxic_side_n_min": max(2, require_int(raw, "toxic_side_n_min")),
        "prior_wr": max(0.40, min(0.70, require_float(raw, "prior_wr"))),
    }
    if override is None:
        _CACHE["policy"] = dict(resolved)
    return resolved


def is_small_sample(n: int, *, policy: dict[str, Any] | None = None) -> bool:
    """True quando N ainda e ruido sob evidence_n_min."""
    cfg = policy if isinstance(policy, dict) else load_sample_size_policy()
    if not bool(cfg.get("enabled", True)):
        return False
    return int(n) < int(cfg["evidence_n_min"])


def is_large_sample(n: int, *, policy: dict[str, Any] | None = None) -> bool:
    """True quando N atinge large_n_min da Lei dos Grandes Numeros."""
    cfg = policy if isinstance(policy, dict) else load_sample_size_policy()
    if not bool(cfg.get("enabled", True)):
        return True
    return int(n) >= int(cfg["large_n_min"])


def sample_reliability(n: int, *, policy: dict[str, Any] | None = None) -> float:
    """Peso empirico n/(n+half_life) no intervalo [0, 1]."""
    cfg = policy if isinstance(policy, dict) else load_sample_size_policy()
    if not bool(cfg.get("enabled", True)):
        return 1.0
    half = float(cfg["reliability_half_life_n"])
    nn = max(0.0, float(n))
    return nn / (nn + half)


def empirical_rate_shrink(
    rate: float,
    *,
    n: int,
    prior: float | None = None,
    policy: dict[str, Any] | None = None,
) -> float:
    """Suaviza taxa empirica em direcao ao prior conforme reliability."""
    cfg = policy if isinstance(policy, dict) else load_sample_size_policy()
    p0 = float(cfg["prior_wr"] if prior is None else prior)
    w = sample_reliability(n, policy=cfg)
    return (1.0 - w) * p0 + w * float(rate)


def binomial_evidence_z(wins: int, n: int, p0: float) -> float:
    """Z-score binomial de wins/n contra probabilidade teorica p0."""
    if n <= 0:
        return 0.0
    p_hat = float(wins) / float(n)
    p = min(max(float(p0), 1e-6), 1.0 - 1e-6)
    se = sqrt(p * (1.0 - p) / float(n))
    if se <= 1e-12:
        return 0.0
    return (p_hat - p) / se


def has_underperformance_evidence(
    wins: int,
    n: int,
    *,
    p0: float,
    policy: dict[str, Any] | None = None,
    min_n: int | None = None,
) -> bool:
    """True quando WR esta significativamente abaixo de p0 com N suficiente."""
    cfg = policy if isinstance(policy, dict) else load_sample_size_policy()
    if not bool(cfg.get("enabled", True)):
        return int(n) > 0 and float(wins) / float(n) + 1e-12 < float(p0)
    threshold = int(min_n) if min_n is not None else int(cfg["evidence_n_min"])
    if int(n) < max(1, threshold):
        return False
    z = binomial_evidence_z(int(wins), int(n), float(p0))
    return z + 1e-12 <= -float(cfg["z_sig_threshold"])


def explore_stake_scale(live_n: int, *, policy: dict[str, Any] | None = None) -> float:
    """Escala de stake EXPLORE: piso no cold-start, 1.0 apos large_n_min."""
    cfg = policy if isinstance(policy, dict) else load_sample_size_policy()
    if not bool(cfg.get("enabled", True)):
        return 1.0
    n = int(live_n)
    if n >= int(cfg["large_n_min"]):
        return 1.0
    floor = float(cfg["explore_stake_scale_floor"])
    rel = sample_reliability(n, policy=cfg)
    return max(floor, min(1.0, floor + (1.0 - floor) * rel))


def attach_sample_size_metrics(metrics: dict[str, Any], live_n: int, *, policy: dict[str, Any] | None = None) -> None:
    """Injeta metricas de tamanho amostral no dicionario de decisao."""
    cfg = policy if isinstance(policy, dict) else load_sample_size_policy()
    n = int(live_n)
    metrics["sample_n"] = n
    metrics["sample_reliability"] = float(sample_reliability(n, policy=cfg))
    metrics["sample_is_small"] = bool(is_small_sample(n, policy=cfg))
    metrics["sample_is_large"] = bool(is_large_sample(n, policy=cfg))
    metrics["explore_stake_scale"] = float(explore_stake_scale(n, policy=cfg))
