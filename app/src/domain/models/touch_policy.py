"""Contrato versionado de treino, replay e execucao Touch/No Touch."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TouchPolicy:
    """Mesmos parametros em ambas as contas; ausencia de evidencia bloqueia compra."""

    symbol: str = "1HZ75V"
    duration_seconds: int = 300
    history_seconds: int = 300
    max_tick_gap_ms: int = 1500
    latency_ms: tuple[int, ...] = (0, 1000, 2000)
    barrier_sigma: tuple[float, ...] = (1.0, 1.5)
    min_edge: float = 0.03
    min_oos_trades: int = 120
    min_train_groups: int = 600
    max_quote_age_ms: int = 2000
    max_stake_pct: float = 0.001
    model_max_age_days: int = 7

    def fingerprint(self) -> str:
        """Invalida artefatos quando mudar o evento, features ou politica de selecao."""
        payload = {"schema": "touch_ticks_v1", **asdict(self)}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def touch_enabled(config: dict) -> bool:
    """Ativacao explicita, sem herdar flags do seletor direcional legado."""
    return config.get("touch", {}).get("enabled") is True


def resolve_touch_policy(config: dict) -> TouchPolicy:
    """Valida knobs antes de treinamento ou qualquer solicitacao de compra."""
    raw = config.get("touch", {})
    values = {key: raw[key] for key in TouchPolicy.__dataclass_fields__ if key in raw}
    for key in ("latency_ms", "barrier_sigma"):
        if key in values:
            values[key] = tuple(values[key])
    policy = TouchPolicy(**values)
    if policy.symbol != "1HZ75V" or policy.duration_seconds != 300:
        raise ValueError("Touch exige 1HZ75V e contrato de 300s")
    if (
        policy.history_seconds < 30
        or policy.max_tick_gap_ms < 1000
        or policy.min_oos_trades < 120
        or policy.min_train_groups < 600
        or not 0 < policy.max_quote_age_ms <= 2000
        or not 0 < policy.max_stake_pct <= 0.01
        or policy.model_max_age_days < 1
        or not math.isfinite(policy.min_edge)
        or policy.min_edge <= 0
        or not policy.latency_ms
        or min(policy.latency_ms) < 0
        or max(policy.latency_ms) < policy.max_quote_age_ms
        or not policy.barrier_sigma
        or any(not math.isfinite(v) or v <= 0 for v in policy.barrier_sigma)
    ):
        raise ValueError("Politica Touch invalida ou piso de evidencia insuficiente")
    return policy
