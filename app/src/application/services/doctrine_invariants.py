"""Invariantes da doutrina LLM/AGENTS carregados do SSOT settings."""

from __future__ import annotations

from typing import Any

from src.domain.config_knobs import load_settings_json, require_bool, require_float, require_int, require_keys


_PRODUCTION_MIN_ACC = 0.53
_CACHE: dict[str, Any] = {"invariants": None}

__all__ = (
    "assert_production_doctrine",
    "load_doctrine_invariants",
    "reset_doctrine_invariants_cache",
)


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


def _loss_hard(settings: dict[str, Any]) -> dict[str, Any]:
    """Le knobs do loss_classifier (FLIP por p_eff no piso hard_p_loss_floor)."""
    infra = settings.get("infra")
    if not isinstance(infra, dict):
        raise ValueError("infra obrigatorio")
    block = infra.get("loss_classifier")
    if not isinstance(block, dict):
        raise ValueError("infra.loss_classifier obrigatorio")
    require_keys(
        block,
        (
            "veto_mode",
            "hard_p_loss_floor",
            "enabled",
            "flip_trust_n",
            "flip_young_shrink",
            "flip_young_p_eff_floor",
        ),
        "infra.loss_classifier",
    )
    return {
        "loss_clf_enabled": require_bool(block, "enabled"),
        "loss_clf_veto_mode": str(block["veto_mode"]).strip().lower(),
        "loss_clf_hard_p_loss_floor": require_float(block, "hard_p_loss_floor"),
        "loss_clf_flip_trust_n": require_int(block, "flip_trust_n"),
        "loss_clf_flip_young_shrink": require_float(block, "flip_young_shrink"),
        "loss_clf_flip_young_p_eff_floor": require_float(block, "flip_young_p_eff_floor"),
    }


def _recovery_timing(settings: dict[str, Any], risk: dict[str, Any]) -> dict[str, Any]:
    """Le recovery cover/kelly floors e timing orchestrator + stop-win."""
    soft = risk["soft_recovery"]
    require_keys(
        soft,
        (
            "amort_cycles_min",
            "amort_cycles_max",
            "cover_multiple",
            "cover_enabled",
            "max_safe_stake_pct_linear3",
        ),
        "risk_management.soft_recovery",
    )
    kelly = risk.get("kelly")
    if not isinstance(kelly, dict):
        raise ValueError("risk_management.kelly obrigatorio")
    require_keys(kelly, ("neutral_bankroll_pct", "min_stake_pct"), "risk_management.kelly")
    orch = settings["orchestrator"]
    require_keys(
        orch,
        (
            "watchdog_stale_tick_seconds",
            "settlement_tolerance_window_seconds",
            "post_settlement_is_trading_wait_seconds",
        ),
        "orchestrator",
    )
    return {
        "amort_cycles_min": int(soft["amort_cycles_min"]),
        "amort_cycles_max": int(soft["amort_cycles_max"]),
        "cover_multiple": require_float(soft, "cover_multiple"),
        "cover_enabled": require_bool(soft, "cover_enabled"),
        "neutral_bankroll_pct": require_float(kelly, "neutral_bankroll_pct"),
        "min_stake_pct": require_float(kelly, "min_stake_pct"),
        "max_safe_stake_pct_linear3": require_float(soft, "max_safe_stake_pct_linear3"),
        "watchdog_stale_tick_seconds": int(orch["watchdog_stale_tick_seconds"]),
        "settlement_tolerance_window_seconds": int(orch["settlement_tolerance_window_seconds"]),
        "post_settlement_is_trading_wait_seconds": int(orch["post_settlement_is_trading_wait_seconds"]),
        "large_account_stop_win_pct": require_float(risk, "large_account_stop_win_pct"),
    }


def load_doctrine_invariants(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """Carrega invariantes tipados da doutrina a partir de settings ou SSOT."""
    use_cache = settings is None
    if use_cache and _CACHE.get("invariants") is not None:
        return dict(_CACHE["invariants"])
    full = settings if isinstance(settings, dict) else load_settings_json()
    execution = _execution_block(full)
    require_keys(
        execution,
        ("force_trade_every_cycle", "mandatory_trade_each_cycle", "sample_size_policy"),
        "orchestrator.execution",
    )
    if "signal_skip" in execution:
        raise ValueError("orchestrator.execution.signal_skip removido da doutrina (gates = tecnico + loss_clf)")
    if "invert_exec_side" in execution:
        raise ValueError("orchestrator.execution.invert_exec_side removido da doutrina")
    risk = full.get("risk_management")
    if not isinstance(risk, dict) or "min_validation_accuracy_gate" not in risk:
        raise ValueError("risk_management.min_validation_accuracy_gate obrigatorio")
    if "large_account_stop_win_pct" not in risk:
        raise ValueError("risk_management.large_account_stop_win_pct obrigatorio")
    dl = full.get("deep_learning")
    if not isinstance(dl, dict) or "online_training" not in dl:
        raise ValueError("deep_learning.online_training obrigatorio")
    cap, pct = _safe_stake(risk)
    resolved: dict[str, Any] = {
        "force_trade_every_cycle": require_bool(execution, "force_trade_every_cycle"),
        "mandatory_trade_each_cycle": require_bool(execution, "mandatory_trade_each_cycle"),
        "online_training": require_bool(dl, "online_training"),
        "min_validation_accuracy_gate": require_float(risk, "min_validation_accuracy_gate"),
        "explore_stake_scale_floor": _explore_floor(execution),
        "max_safe_stake_cap": float(cap),
        "max_safe_stake_pct": float(pct),
        **_loss_hard(full),
        **_recovery_timing(full, risk),
    }
    if use_cache:
        _CACHE["invariants"] = dict(resolved)
    return resolved


def assert_production_doctrine(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """Valida settings de producao contra pisos da doutrina AGENTS; retorna invariantes."""
    inv = load_doctrine_invariants(settings)
    if inv["force_trade_every_cycle"]:
        raise ValueError("force_trade_every_cycle deve ser false na doutrina de producao")
    if inv["mandatory_trade_each_cycle"]:
        raise ValueError("mandatory_trade_each_cycle deve ser false na doutrina de producao")
    if inv["online_training"]:
        raise ValueError("online_training deve ser false na doutrina de producao")
    if inv["loss_clf_veto_mode"] != "hard":
        raise ValueError("loss_classifier.veto_mode deve ser hard")
    if abs(float(inv["loss_clf_hard_p_loss_floor"]) - 0.55) > 1e-9:
        raise ValueError("loss_classifier.hard_p_loss_floor deve ser 0.55")
    if int(inv["loss_clf_flip_trust_n"]) != 32:
        raise ValueError("loss_classifier.flip_trust_n deve ser 32")
    if abs(float(inv["loss_clf_flip_young_shrink"]) - 0.35) > 1e-9:
        raise ValueError("loss_classifier.flip_young_shrink deve ser 0.35")
    if abs(float(inv["loss_clf_flip_young_p_eff_floor"]) - 0.55) > 1e-9:
        raise ValueError("loss_classifier.flip_young_p_eff_floor deve ser 0.55")
    if not inv["loss_clf_enabled"]:
        raise ValueError("loss_classifier.enabled deve ser true")
    if int(inv["watchdog_stale_tick_seconds"]) != 300:
        raise ValueError("watchdog_stale_tick_seconds deve ser 300")
    if int(inv["settlement_tolerance_window_seconds"]) != 600:
        raise ValueError("settlement_tolerance_window_seconds deve ser 600")
    if int(inv["post_settlement_is_trading_wait_seconds"]) != 90:
        raise ValueError("post_settlement_is_trading_wait_seconds deve ser 90")
    if int(inv["amort_cycles_min"]) < 1 or int(inv["amort_cycles_max"]) > 4:
        raise ValueError("amort_cycles deve estar entre 1 e 4")
    if float(inv["cover_multiple"]) < 1.0 or float(inv["cover_multiple"]) > 2.0:
        raise ValueError("cover_multiple deve estar em [1.0, 2.0]")
    if inv["cover_enabled"] is not False:
        raise ValueError("cover_enabled deve ser false (sem amortizacao em massa)")
    if abs(float(inv["neutral_bankroll_pct"]) - 0.01) > 1e-9:
        raise ValueError("neutral_bankroll_pct deve ser 0.01")
    if abs(float(inv["min_stake_pct"]) - 0.01) > 1e-9:
        raise ValueError("min_stake_pct deve ser 0.01")
    if abs(float(inv["max_safe_stake_pct_linear3"]) - 0.025) > 1e-9:
        raise ValueError("max_safe_stake_pct_linear3 deve ser 0.025")
    if float(inv["large_account_stop_win_pct"]) <= 0.0 or float(inv["large_account_stop_win_pct"]) > 5.0:
        raise ValueError("large_account_stop_win_pct deve estar em (0.0, 5.0]")
    if float(inv["min_validation_accuracy_gate"]) + 1e-12 < _PRODUCTION_MIN_ACC:
        raise ValueError(f"min_validation_accuracy_gate < {_PRODUCTION_MIN_ACC}")
    if abs(float(inv["explore_stake_scale_floor"]) - 0.40) > 1e-9:
        raise ValueError("explore_stake_scale_floor deve ser 0.40")
    if float(inv["max_safe_stake_cap"]) <= 0.0 or float(inv["max_safe_stake_pct"]) <= 0.0:
        raise ValueError("max_safe_stake_cap/pct devem ser > 0")
    return inv
