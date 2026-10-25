"""Invariantes da doutrina LLM/AGENTS carregados do SSOT settings."""

from __future__ import annotations

from typing import Any

from src.domain.config_knobs import load_settings_json, require_bool, require_float, require_keys


_PRODUCTION_MIN_ACC = 0.50
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
    require_keys(
        execution,
        (
            "invert_exec_side",
            "skip_exec_vs_candle",
            "skip_below_soft_min_acc",
            "skip_scale_candle_discord",
            "skip_neg_edge",
            "skip_doji",
            "counter_trend_min_edge",
        ),
        "orchestrator.execution",
    )
    risk = full.get("risk_management")
    if not isinstance(risk, dict) or "min_validation_accuracy_gate" not in risk:
        raise ValueError("risk_management.min_validation_accuracy_gate obrigatorio")
    if "large_account_stop_win_pct" not in risk:
        raise ValueError("risk_management.large_account_stop_win_pct obrigatorio")
    dl = full.get("deep_learning")
    if not isinstance(dl, dict) or "online_training" not in dl:
        raise ValueError("deep_learning.online_training obrigatorio")
    infra = full.get("infra")
    if not isinstance(infra, dict):
        raise ValueError("infra obrigatorio")
    cap, pct = _safe_stake(risk)
    resolved: dict[str, Any] = {
        "force_trade_every_cycle": require_bool(execution, "force_trade_every_cycle"),
        "mandatory_trade_each_cycle": require_bool(execution, "mandatory_trade_each_cycle"),
        "invert_exec_side": require_bool(execution, "invert_exec_side"),
        "skip_exec_vs_candle": require_bool(execution, "skip_exec_vs_candle"),
        "skip_below_soft_min_acc": require_bool(execution, "skip_below_soft_min_acc"),
        "skip_scale_candle_discord": require_bool(execution, "skip_scale_candle_discord"),
        "skip_neg_edge": require_bool(execution, "skip_neg_edge"),
        "skip_doji": require_bool(execution, "skip_doji"),
        "counter_trend_min_edge": require_float(execution, "counter_trend_min_edge"),
        "online_training": require_bool(dl, "online_training"),
        "min_validation_accuracy_gate": require_float(risk, "min_validation_accuracy_gate"),
        "explore_stake_scale_floor": _explore_floor(execution),
        "max_safe_stake_cap": float(cap),
        "max_safe_stake_pct": float(pct),
        **_recovery_timing(full, risk),
    }
    if use_cache:
        _CACHE["invariants"] = dict(resolved)
    return resolved


def _eq_int(inv: dict[str, Any], key: str, expected: int, msg: str) -> None:
    """Fail-closed se inteiro divergir."""
    if int(inv[key]) != expected:
        raise ValueError(msg)


def _eq_float(inv: dict[str, Any], key: str, expected: float, msg: str) -> None:
    """Fail-closed se float divergir."""
    if abs(float(inv[key]) - expected) > 1e-9:
        raise ValueError(msg)


def _eq_bool(inv: dict[str, Any], key: str, *, expected: bool, msg: str) -> None:
    """Fail-closed se bool divergir."""
    if bool(inv[key]) is not expected:
        raise ValueError(msg)


def assert_production_doctrine(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """Valida settings de producao contra pisos da doutrina AGENTS; retorna invariantes."""
    inv = load_doctrine_invariants(settings)
    full = settings if isinstance(settings, dict) else load_settings_json()
    execution = _execution_block(full)
    require_keys(
        execution,
        ("force_trade_every_cycle", "mandatory_trade_each_cycle"),
        "orchestrator.execution",
    )
    if "cal_soft_edge_skip_enabled" in execution or "cal_soft_edge_margin_floor" in execution:
        raise ValueError("cal_soft_edge_* removido da doutrina (sem trava por Edge)")
    scale = execution.get("scale_vision")
    if not isinstance(scale, dict):
        raise ValueError("scale_vision ausente")
    if require_bool(scale, "adapt_retract_enabled"):
        raise ValueError("scale_vision.adapt_retract_enabled deve ser false (direcao = TCN + loss-clf)")
    if not require_bool(scale, "adapt_tape_require_strong"):
        raise ValueError("scale_vision.adapt_tape_require_strong deve ser true")
    if abs(float(require_float(scale, "adapt_explos_max_tcn_edge")) - 0.05) > 1e-9:
        raise ValueError("scale_vision.adapt_explos_max_tcn_edge deve ser 0.05")
    if "adapt_soft_margin" in scale:
        raise ValueError("scale_vision.adapt_soft_margin removido (retract adapta sem freio de margem)")
    for key, msg in (
        ("force_trade_every_cycle", "force_trade_every_cycle deve ser false na doutrina de producao"),
        ("mandatory_trade_each_cycle", "mandatory_trade_each_cycle deve ser false na doutrina de producao"),
        ("online_training", "online_training deve ser false na doutrina de producao"),
        ("invert_exec_side", "invert_exec_side deve ser false na doutrina de producao"),
        ("skip_below_soft_min_acc", "skip_below_soft_min_acc deve ser false na doutrina de producao"),
        ("skip_exec_vs_candle", "skip_exec_vs_candle deve ser false (vela nao decide lado)"),
        ("skip_scale_candle_discord", "skip_scale_candle_discord deve ser false (sem SCALE adapt)"),
        ("skip_doji", "skip_doji deve ser false (vela nao decide lado)"),
        ("skip_neg_edge", "skip_neg_edge deve ser false na doutrina de producao"),
    ):
        _eq_bool(inv, key, expected=False, msg=msg)
    _eq_bool(inv, "cover_enabled", expected=True, msg="cover_enabled deve ser true (cover soft capped)")
    _eq_int(inv, "watchdog_stale_tick_seconds", 300, "watchdog_stale_tick_seconds deve ser 300")
    _eq_int(inv, "settlement_tolerance_window_seconds", 600, "settlement_tolerance_window_seconds deve ser 600")
    _eq_int(inv, "post_settlement_is_trading_wait_seconds", 90, "post_settlement_is_trading_wait_seconds deve ser 90")
    _eq_float(inv, "counter_trend_min_edge", 0.08, "counter_trend_min_edge deve ser 0.08")
    if int(inv["amort_cycles_min"]) != 1 or int(inv["amort_cycles_max"]) != 1:
        raise ValueError("amort_cycles_min/max devem ser 1 (cover_l0 zera PEND em 1 WIN sob L0)")
    _eq_float(inv, "max_safe_stake_pct", 0.035, "max_safe_stake_pct L0 deve ser 0.035 (cover amort<=1)")
    _eq_float(inv, "cover_multiple", 1.0, "cover_multiple deve ser 1.0")
    _eq_float(inv, "neutral_bankroll_pct", 0.01, "neutral_bankroll_pct deve ser 0.01")
    _eq_float(inv, "min_stake_pct", 0.01, "min_stake_pct deve ser 0.01")
    _eq_float(inv, "max_safe_stake_pct_linear3", 0.025, "max_safe_stake_pct_linear3 deve ser 0.025")
    if float(inv["large_account_stop_win_pct"]) <= 0.0 or float(inv["large_account_stop_win_pct"]) > 5.0:
        raise ValueError("large_account_stop_win_pct deve estar em (0.0, 5.0]")
    if float(inv["min_validation_accuracy_gate"]) + 1e-12 < _PRODUCTION_MIN_ACC:
        raise ValueError(f"min_validation_accuracy_gate < {_PRODUCTION_MIN_ACC}")
    _eq_float(inv, "explore_stake_scale_floor", 0.40, "explore_stake_scale_floor deve ser 0.40")
    if float(inv["max_safe_stake_cap"]) <= 0.0 or float(inv["max_safe_stake_pct"]) <= 0.0:
        raise ValueError("max_safe_stake_cap/pct devem ser > 0")
    return inv
