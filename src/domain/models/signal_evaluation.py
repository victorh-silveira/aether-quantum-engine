"""Value objects imutaveis de avaliacao de sinal e decisao de stake."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


def _to_float(value: Any, default: float = 0.0) -> float:
    """Converte valor opcional para float com fallback seguro."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_opt_float(value: Any) -> float | None:
    """Converte valor opcional para float ou None."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class SignalEvaluation:
    """Visao imutavel das metricas de sinal consumidas pelo dominio de risco."""

    execute: bool = True
    deploy_ok: bool | None = None
    signal_status: str = ""
    meta_veto_mode: str = ""
    senior_trader_conviction: float = 0.0
    stake_regime: str = "EXPLORE"
    live_n: int = 0
    live_wr: float | None = None
    explore_stake_scale: float | None = None
    kelly_fraction_scale: float = 1.0
    call_votes: int = 0
    put_votes: int = 0
    val_accuracy: float = 0.0
    trade_score: float = 0.0
    conviction: float = 0.0
    calibrated_prob: float = 0.0
    execution_candidate_ready: bool | None = None
    scale_force_explore: bool = False
    recovery_force_explore: bool = False
    recovery_forced: bool = False
    recovery_infeasible: bool = False
    recovery_cap_mode: str = ""
    raw: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    @classmethod
    def from_metrics(cls, metrics: Mapping[str, Any] | SignalEvaluation | None) -> SignalEvaluation:
        """Constroi SignalEvaluation imutavel a partir de dicionario ou instancia existente."""
        if isinstance(metrics, cls):
            return metrics
        if not isinstance(metrics, Mapping):
            return cls()
        frozen_raw = MappingProxyType(dict(metrics))
        deploy_raw = metrics.get("deploy_ok")
        ready_raw = metrics.get("execution_candidate_ready")
        return cls(
            execute=bool(metrics.get("execute", True)),
            deploy_ok=bool(deploy_raw) if deploy_raw is not None else None,
            signal_status=str(metrics.get("signal_status") or "").strip(),
            meta_veto_mode=str(metrics.get("meta_veto_mode") or "").strip(),
            senior_trader_conviction=_to_float(metrics.get("senior_trader_conviction"), 0.0),
            stake_regime=str(metrics.get("stake_regime") or "EXPLORE").upper(),
            live_n=int(_to_float(metrics.get("live_n"), 0.0)),
            live_wr=_to_opt_float(metrics.get("live_wr")),
            explore_stake_scale=_to_opt_float(metrics.get("explore_stake_scale")),
            kelly_fraction_scale=_to_float(metrics.get("kelly_fraction_scale"), 1.0),
            call_votes=int(_to_float(metrics.get("call_votes"), 0.0)),
            put_votes=int(_to_float(metrics.get("put_votes"), 0.0)),
            val_accuracy=_to_float(metrics.get("val_accuracy"), 0.0),
            trade_score=_to_float(metrics.get("trade_score"), 0.0),
            conviction=_to_float(metrics.get("conviction"), 0.0),
            calibrated_prob=_to_float(metrics.get("calibrated_prob"), 0.0),
            execution_candidate_ready=bool(ready_raw) if ready_raw is not None else None,
            scale_force_explore=bool(metrics.get("scale_force_explore")),
            recovery_force_explore=bool(metrics.get("recovery_force_explore")),
            recovery_forced=bool(metrics.get("recovery_forced")),
            recovery_infeasible=bool(metrics.get("recovery_infeasible")),
            recovery_cap_mode=str(metrics.get("recovery_cap_mode") or ""),
            raw=frozen_raw,
        )

    def get(self, key: str, default: Any = None) -> Any:
        """Leitura segura e imutavel por chave para compatibilidade com helpers de risco."""
        return self.raw.get(key, default)

    def __contains__(self, key: object) -> bool:
        """Suporta verificacao 'chave in evaluation' de forma somente leitura."""
        return key in self.raw

    def __getitem__(self, key: str) -> Any:
        """Suporta leitura imutavel por subscript evaluation['chave']."""
        return self.raw[key]

    def to_dict(self) -> dict[str, Any]:
        """Retorna copia rasa mutavel apenas para fronteira de aplicacao/telemetria."""
        return dict(self.raw)


@dataclass(frozen=True)
class StakeDecision:
    """Resultado imutavel do calculo de stake e telemetria associada."""

    stake: float
    f_star: float = 0.0
    stake_regime: str = "EXPLORE"
    kelly_fraction_scale: float = 1.0
    explore_stake_scale: float | None = None
    consensus_entropy_retention: float = 1.0
    super_concordance_booster_active: bool = False
    super_concordance_fraction_multiplier: float | None = None
    kelly_fraction_effective: float | None = None
    provisional_stake_cap_applied: bool = False
    extra_telemetry: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def to_telemetry_updates(self) -> dict[str, Any]:
        """Retorna campos de telemetria calculados para aplicacao fora do dominio."""
        updates: dict[str, Any] = dict(self.extra_telemetry)
        updates.update(
            {
                "f_star": self.f_star,
                "kelly_fraction": self.f_star,
                "stake_regime": self.stake_regime,
                "kelly_fraction_scale": self.kelly_fraction_scale,
                "consensus_entropy_retention": self.consensus_entropy_retention,
            }
        )
        if self.explore_stake_scale is not None:
            updates["explore_stake_scale"] = self.explore_stake_scale
        if self.super_concordance_booster_active:
            updates["super_concordance_booster_active"] = True
            updates["super_concordance_fraction_multiplier"] = self.super_concordance_fraction_multiplier
            updates["kelly_fraction_effective"] = self.kelly_fraction_effective
        if self.provisional_stake_cap_applied:
            updates["provisional_stake_cap_applied"] = True
        return updates
