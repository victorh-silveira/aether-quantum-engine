"""Politica unica de pisos e flags de risco/execucao lida da config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RiskPolicy:
    """Snapshot imutavel dos pisos e limites de risco usados no ciclo."""

    mandatory_trade_each_cycle: bool
    require_meta_for_execution: bool
    require_triton_for_execution: bool
    max_stake_pct: float
    max_bankroll_stake_fraction: float
    recovery_min_trade_score: float
    recovery_min_val_accuracy: float
    recovery_min_hurst: float
    min_meta_payoff_zscore: float
    deploy_gate_enabled: bool


def load_risk_policy(config: dict[str, Any] | None) -> RiskPolicy:
    """Extrai RiskPolicy a partir de settings.json / config runtime."""
    cfg = config if isinstance(config, dict) else {}
    orch = cfg.get("orchestrator", {}) if isinstance(cfg.get("orchestrator"), dict) else {}
    exec_cfg = orch.get("execution", {}) if isinstance(orch.get("execution"), dict) else {}
    quality = exec_cfg.get("quality_gate", {}) if isinstance(exec_cfg.get("quality_gate"), dict) else {}
    loss_protection = exec_cfg.get("loss_protection", {}) if isinstance(exec_cfg.get("loss_protection"), dict) else {}
    risk = cfg.get("risk_management", {}) if isinstance(cfg.get("risk_management"), dict) else {}
    kelly = risk.get("kelly", {}) if isinstance(risk.get("kelly"), dict) else {}
    dl = cfg.get("deep_learning", {}) if isinstance(cfg.get("deep_learning"), dict) else {}
    deploy = dl.get("deploy_gate", {}) if isinstance(dl.get("deploy_gate"), dict) else {}
    infra = cfg.get("infra", {}) if isinstance(cfg.get("infra"), dict) else {}
    triton = infra.get("triton", {}) if isinstance(infra.get("triton"), dict) else {}
    require_triton = bool(triton.get("require_for_execution", exec_cfg.get("require_triton_for_execution", False)))
    return RiskPolicy(
        mandatory_trade_each_cycle=bool(exec_cfg.get("mandatory_trade_each_cycle", False)),
        require_meta_for_execution=bool(exec_cfg.get("require_meta_for_execution", True)),
        require_triton_for_execution=require_triton,
        max_stake_pct=float(kelly.get("max_stake_pct", 0.035)),
        max_bankroll_stake_fraction=float(kelly.get("max_bankroll_stake_fraction", 0.035)),
        recovery_min_trade_score=float(kelly.get("recovery_min_trade_score", 0.64)),
        recovery_min_val_accuracy=float(kelly.get("recovery_min_val_accuracy", 0.62)),
        recovery_min_hurst=float(loss_protection.get("recovery_min_hurst", 0.50)),
        min_meta_payoff_zscore=float(quality.get("min_meta_payoff_zscore", 0.5)),
        deploy_gate_enabled=bool(deploy.get("enabled", False)),
    )


def validate_engine_risk_config(config: dict[str, Any]) -> list[str]:
    """Retorna lista de inconsistencias de risco/execucao na config."""
    policy = load_risk_policy(config)
    errors: list[str] = []
    if policy.max_stake_pct <= 0.0 or policy.max_stake_pct > 0.10:
        errors.append(f"kelly.max_stake_pct fora de (0, 0.10]: {policy.max_stake_pct}")
    if policy.max_bankroll_stake_fraction <= 0.0 or policy.max_bankroll_stake_fraction > 0.10:
        errors.append(f"kelly.max_bankroll_stake_fraction fora de (0, 0.10]: {policy.max_bankroll_stake_fraction}")
    if policy.max_bankroll_stake_fraction + 1e-12 < policy.max_stake_pct:
        errors.append("max_bankroll_stake_fraction < max_stake_pct")
    if policy.mandatory_trade_each_cycle and not policy.deploy_gate_enabled:
        errors.append("mandatory_trade_each_cycle=true exige deep_learning.deploy_gate.enabled=true")
    if policy.recovery_min_trade_score < 0.45 or policy.recovery_min_trade_score > 0.95:
        errors.append(f"recovery_min_trade_score suspeito: {policy.recovery_min_trade_score}")
    return errors
