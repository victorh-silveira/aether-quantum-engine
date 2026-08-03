"""Invariantes da doutrina LLM/AGENTS carregados do SSOT settings."""

from __future__ import annotations

from typing import Any

from src.domain.config_knobs import load_settings_json, require_bool, require_float, require_keys


_PRODUCTION_MIN_HARD_CAL = 0.05
_PRODUCTION_MIN_ACC = 0.53
_PRODUCTION_MIN_EDGE = 0.0

_CACHE: dict[str, Any] = {"invariants": None}


def reset_doctrine_invariants_cache() -> None:
    """Limpa cache dos invariantes da doutrina."""
    _CACHE["invariants"] = None


def _execution_block(settings: dict[str, Any]) -> dict[str, Any]:
    """Extrai orchestrator.execution tipado do settings raiz."""
    orch = settings.get("orchestrator")
    if not isinstance(orch, dict):
        raise ValueError("orchestrator obrigatorio")
    execution = orch.get("execution")
    if not isinstance(execution, dict):
        raise ValueError("orchestrator.execution obrigatorio")
    return execution


def _quality_edges(execution: dict[str, Any]) -> tuple[float, float]:
    """Le min_payoff_edge top-level e regular do quality_gate."""
    qg = execution.get("quality_gate")
    if not isinstance(qg, dict):
        raise ValueError("orchestrator.execution.quality_gate obrigatorio")
    require_keys(qg, ("min_payoff_edge", "regular"), "orchestrator.execution.quality_gate")
    top = require_float(qg, "min_payoff_edge")
    regular = qg.get("regular")
    if not isinstance(regular, dict) or "min_payoff_edge" not in regular:
        raise ValueError("orchestrator.execution.quality_gate.regular.min_payoff_edge obrigatorio")
    return top, require_float(regular, "min_payoff_edge")


def _explore_floor(execution: dict[str, Any]) -> float:
    """Le explore_stake_scale_floor de sample_size_policy."""
    ssp = execution.get("sample_size_policy")
    if not isinstance(ssp, dict) or "explore_stake_scale_floor" not in ssp:
        raise ValueError("orchestrator.execution.sample_size_policy.explore_stake_scale_floor obrigatorio")
    return float(ssp["explore_stake_scale_floor"])


def _safe_stake(risk: dict[str, Any]) -> tuple[float, float]:
    """Le max_safe_stake_cap e max_safe_stake_pct do soft_recovery."""
    soft = risk.get("soft_recovery")
    if not isinstance(soft, dict):
        raise ValueError("risk_management.soft_recovery obrigatorio")
    require_keys(soft, ("max_safe_stake_cap", "max_safe_stake_pct"), "risk_management.soft_recovery")
    return require_float(soft, "max_safe_stake_cap"), require_float(soft, "max_safe_stake_pct")


def load_doctrine_invariants(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """Carrega invariantes tipados da doutrina a partir de settings ou SSOT."""
    use_cache = settings is None
    if use_cache and _CACHE.get("invariants") is not None:
        return dict(_CACHE["invariants"])
    full = settings if isinstance(settings, dict) else load_settings_json()
    execution = _execution_block(full)
    require_keys(
        execution,
        ("hard_cal_margin_floor", "force_trade_every_cycle", "quality_gate", "sample_size_policy"),
        "orchestrator.execution",
    )
    risk = full.get("risk_management")
    if not isinstance(risk, dict) or "min_validation_accuracy_gate" not in risk:
        raise ValueError("risk_management.min_validation_accuracy_gate obrigatorio")
    edge_top, edge_regular = _quality_edges(execution)
    cap, pct = _safe_stake(risk)
    resolved = {
        "force_trade_every_cycle": require_bool(execution, "force_trade_every_cycle"),
        "hard_cal_margin_floor": require_float(execution, "hard_cal_margin_floor"),
        "min_payoff_edge": float(edge_top),
        "regular_min_payoff_edge": float(edge_regular),
        "min_validation_accuracy_gate": require_float(risk, "min_validation_accuracy_gate"),
        "explore_stake_scale_floor": _explore_floor(execution),
        "max_safe_stake_cap": float(cap),
        "max_safe_stake_pct": float(pct),
    }
    if use_cache:
        _CACHE["invariants"] = dict(resolved)
    return resolved


def resolve_hard_cal_margin_floor(exec_cfg: dict[str, Any] | None) -> float:
    """Resolve piso calibrado duro: override em exec_cfg ou SSOT da doutrina."""
    cfg = exec_cfg if isinstance(exec_cfg, dict) else {}
    if "hard_cal_margin_floor" in cfg:
        raw = cfg.get("hard_cal_margin_floor")
        return float(raw or 0.0)
    return float(load_doctrine_invariants()["hard_cal_margin_floor"])


def assert_production_doctrine(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """Valida settings de producao contra pisos da doutrina AGENTS; retorna invariantes."""
    inv = load_doctrine_invariants(settings)
    if inv["force_trade_every_cycle"]:
        raise ValueError("force_trade_every_cycle deve ser false na doutrina de producao")
    if float(inv["hard_cal_margin_floor"]) + 1e-12 < _PRODUCTION_MIN_HARD_CAL:
        raise ValueError(f"hard_cal_margin_floor < {_PRODUCTION_MIN_HARD_CAL}")
    if float(inv["min_payoff_edge"]) + 1e-12 < _PRODUCTION_MIN_EDGE:
        raise ValueError("min_payoff_edge negativo na doutrina de producao")
    if float(inv["regular_min_payoff_edge"]) + 1e-12 < _PRODUCTION_MIN_EDGE:
        raise ValueError("regular.min_payoff_edge negativo na doutrina de producao")
    if float(inv["min_validation_accuracy_gate"]) + 1e-12 < _PRODUCTION_MIN_ACC:
        raise ValueError(f"min_validation_accuracy_gate < {_PRODUCTION_MIN_ACC}")
    if float(inv["explore_stake_scale_floor"]) <= 0.0:
        raise ValueError("explore_stake_scale_floor deve ser > 0")
    if float(inv["max_safe_stake_cap"]) <= 0.0 or float(inv["max_safe_stake_pct"]) <= 0.0:
        raise ValueError("max_safe_stake_cap/pct devem ser > 0")
    return inv
