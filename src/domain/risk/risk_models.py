"""Modelos, politica de risco, cooldown de proposta e finalizacao de cluster."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.domain.risk.executed_stake_reconciliation import apply_contract_settlement_result
from src.domain.risk.risk_recovery_state import apply_cluster_profit_to_recovery_state


__all__ = (
    "NullLogger",
    "ProposalSkipMixin",
    "RiskPolicy",
    "apply_contract_settlement_result",
    "finalize_risk_cluster",
    "load_risk_policy",
    "validate_engine_risk_config",
)


class NullLogger:
    """Logger noop de dominio quando nenhum adaptador externo e injetado."""

    def debug(self, *_args: Any, **_kwargs: Any) -> None:
        """Ignora mensagens debug no dominio puro."""

    def info(self, *_args: Any, **_kwargs: Any) -> None:
        """Ignora mensagens info no dominio puro."""


def finalize_risk_cluster(risk_manager: Any) -> None:
    """Atualiza perdas consecutivas e limpa estado do cluster."""
    cluster_profit = sum(risk_manager.cluster_results.values())
    apply_cluster_profit_to_recovery_state(risk_manager, cluster_profit)
    risk_manager.active_contract_ids = []
    risk_manager.contract_to_symbol = {}
    risk_manager.cluster_results = {}
    risk_manager.expected_cluster_settlements = 0


class ProposalSkipMixin:
    """Mixin com controle de skip temporario por simbolo."""

    proposal_skip_cycles: dict[str, int]

    def register_proposal_failure(self, symbol: str, *, cycles: int = 6) -> None:
        """Marca simbolo para pular selecao apos falha de proposta na Deriv."""
        hold = max(int(cycles), int(self.proposal_skip_cycles.get(str(symbol), 0)))
        self.proposal_skip_cycles[str(symbol)] = hold

    def decay_proposal_skip_cycles(self) -> None:
        """Reduz cooldown de simbolos com proposta rejeitada."""
        expired: list[str] = []
        for symbol, remaining in self.proposal_skip_cycles.items():
            left = int(remaining) - 1
            if left <= 0:
                expired.append(symbol)
            else:
                self.proposal_skip_cycles[symbol] = left
        for symbol in expired:
            self.proposal_skip_cycles.pop(symbol, None)

    def proposal_skip_symbols(self) -> frozenset[str]:
        """Simbolos temporariamente excluidos apos falha de proposta."""
        return frozenset(s for s, n in self.proposal_skip_cycles.items() if int(n) > 0)


@dataclass(frozen=True)
class RiskPolicy:
    """Snapshot imutavel dos pisos e limites de risco usados no ciclo."""

    mandatory_trade_each_cycle: bool
    require_meta_for_execution: bool
    max_stake_pct: float
    max_bankroll_stake_fraction: float
    recovery_min_trade_score: float
    recovery_min_val_accuracy: float
    recovery_min_hurst: float
    min_meta_payoff_zscore: float


def load_risk_policy(config: dict[str, Any] | None) -> RiskPolicy:
    """Extrai RiskPolicy a partir de settings.json / config runtime."""
    cfg = config if isinstance(config, dict) else {}
    orch = cfg.get("orchestrator", {}) if isinstance(cfg.get("orchestrator"), dict) else {}
    exec_cfg = orch.get("execution", {}) if isinstance(orch.get("execution"), dict) else {}
    quality = exec_cfg.get("quality_gate", {}) if isinstance(exec_cfg.get("quality_gate"), dict) else {}
    loss_protection = exec_cfg.get("loss_protection", {}) if isinstance(exec_cfg.get("loss_protection"), dict) else {}
    risk = cfg.get("risk_management", {}) if isinstance(cfg.get("risk_management"), dict) else {}
    kelly = risk.get("kelly", {}) if isinstance(risk.get("kelly"), dict) else {}
    return RiskPolicy(
        mandatory_trade_each_cycle=bool(exec_cfg.get("mandatory_trade_each_cycle", False)),
        require_meta_for_execution=bool(exec_cfg.get("require_meta_for_execution", True)),
        max_stake_pct=float(kelly.get("max_stake_pct", 0.035)),
        max_bankroll_stake_fraction=float(kelly.get("max_bankroll_stake_fraction", 0.035)),
        recovery_min_trade_score=float(kelly.get("recovery_min_trade_score", 0.64)),
        recovery_min_val_accuracy=float(kelly.get("recovery_min_val_accuracy", 0.62)),
        recovery_min_hurst=float(loss_protection.get("recovery_min_hurst", 0.50)),
        min_meta_payoff_zscore=float(quality.get("min_meta_payoff_zscore", 0.5)),
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
    if policy.recovery_min_trade_score < 0.45 or policy.recovery_min_trade_score > 0.95:
        errors.append(f"recovery_min_trade_score suspeito: {policy.recovery_min_trade_score}")
    return errors
