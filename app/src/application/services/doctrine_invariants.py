"""Invariantes da doutrina LLM/AGENTS carregados do SSOT settings."""

from __future__ import annotations

from typing import Any

from src.domain.config_knobs import load_settings_json, require_bool, require_float, require_keys


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


def _loss_flip(settings: dict[str, Any]) -> dict[str, Any]:
    """Le knobs flip_* do loss_classifier."""
    infra = settings.get("infra")
    if not isinstance(infra, dict):
        raise ValueError("infra obrigatorio")
    block = infra.get("loss_classifier")
    if not isinstance(block, dict):
        raise ValueError("infra.loss_classifier obrigatorio")
    require_keys(
        block,
        ("flip_require_auto_learn", "flip_seed_waive_edge_min"),
        "infra.loss_classifier",
    )
    return {
        "flip_require_auto_learn": require_bool(block, "flip_require_auto_learn"),
        "flip_seed_waive_edge_min": require_float(block, "flip_seed_waive_edge_min"),
    }


def _fusion_and_skip(execution: dict[str, Any]) -> dict[str, Any]:
    """Le fusion_block e neg_edge_deep do execution."""
    scale = execution.get("scale_vision")
    if not isinstance(scale, dict) or "fusion_block_when_tcn_pos_edge" not in scale:
        raise ValueError("orchestrator.execution.scale_vision.fusion_block_when_tcn_pos_edge obrigatorio")
    require_keys(
        scale,
        (
            "fusion_block_when_tcn_candle_agree",
            "fusion_loss_requires_auto_learn",
            "fusion_loss_seed_weight_mult",
        ),
        "orchestrator.execution.scale_vision",
    )
    skip = execution.get("signal_skip")
    if not isinstance(skip, dict) or "neg_edge_deep_edge_floor" not in skip:
        raise ValueError("orchestrator.execution.signal_skip.neg_edge_deep_edge_floor obrigatorio")
    return {
        "fusion_block_when_tcn_pos_edge": require_bool(scale, "fusion_block_when_tcn_pos_edge"),
        "fusion_block_when_tcn_candle_agree": require_bool(scale, "fusion_block_when_tcn_candle_agree"),
        "fusion_loss_requires_auto_learn": require_bool(scale, "fusion_loss_requires_auto_learn"),
        "fusion_loss_seed_weight_mult": require_float(scale, "fusion_loss_seed_weight_mult"),
        "neg_edge_deep_edge_floor": require_float(skip, "neg_edge_deep_edge_floor"),
    }


def _recovery_timing(settings: dict[str, Any], risk: dict[str, Any]) -> dict[str, Any]:
    """Le recovery amort/cover e timing orchestrator + stop-win."""
    soft = risk["soft_recovery"]
    require_keys(
        soft,
        ("amort_cycles_min", "amort_cycles_max", "cover_multiple", "max_safe_stake_pct_linear3"),
        "risk_management.soft_recovery",
    )
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
        ("force_trade_every_cycle", "mandatory_trade_each_cycle", "invert_exec_side", "sample_size_policy"),
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
    cap, pct = _safe_stake(risk)
    resolved: dict[str, Any] = {
        "force_trade_every_cycle": require_bool(execution, "force_trade_every_cycle"),
        "mandatory_trade_each_cycle": require_bool(execution, "mandatory_trade_each_cycle"),
        "invert_exec_side": require_bool(execution, "invert_exec_side"),
        "online_training": require_bool(dl, "online_training"),
        "min_validation_accuracy_gate": require_float(risk, "min_validation_accuracy_gate"),
        "explore_stake_scale_floor": _explore_floor(execution),
        "max_safe_stake_cap": float(cap),
        "max_safe_stake_pct": float(pct),
        "signal_skip_enabled": False,
        "signal_skip_min_direction_margin": None,
        **_loss_flip(full),
        **_fusion_and_skip(execution),
        **_recovery_timing(full, risk),
    }
    signal_skip = execution.get("signal_skip")
    if isinstance(signal_skip, dict) and "enabled" in signal_skip:
        resolved["signal_skip_enabled"] = require_bool(signal_skip, "enabled")
        if resolved["signal_skip_enabled"]:
            require_keys(signal_skip, ("min_direction_margin",), "orchestrator.execution.signal_skip")
            resolved["signal_skip_min_direction_margin"] = require_float(signal_skip, "min_direction_margin")
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
    if inv["invert_exec_side"]:
        raise ValueError("invert_exec_side deve ser false na doutrina de producao")
    if inv["online_training"]:
        raise ValueError("online_training deve ser false na doutrina de producao")
    if not inv["flip_require_auto_learn"]:
        raise ValueError("flip_require_auto_learn deve ser true na doutrina de producao")
    if abs(float(inv["flip_seed_waive_edge_min"]) + 0.08) > 1e-9:
        raise ValueError("flip_seed_waive_edge_min deve ser -0.08")
    if not inv["fusion_block_when_tcn_pos_edge"]:
        raise ValueError("fusion_block_when_tcn_pos_edge deve ser true")
    if not inv["fusion_block_when_tcn_candle_agree"]:
        raise ValueError("fusion_block_when_tcn_candle_agree deve ser true")
    if not inv["fusion_loss_requires_auto_learn"]:
        raise ValueError("fusion_loss_requires_auto_learn deve ser true")
    if float(inv["fusion_loss_seed_weight_mult"]) > 0.15 + 1e-12:
        raise ValueError("fusion_loss_seed_weight_mult deve ser <= 0.15")
    if abs(float(inv["neg_edge_deep_edge_floor"]) + 0.12) > 1e-9:
        raise ValueError("neg_edge_deep_edge_floor deve ser -0.12")
    if int(inv["watchdog_stale_tick_seconds"]) != 300:
        raise ValueError("watchdog_stale_tick_seconds deve ser 300")
    if int(inv["settlement_tolerance_window_seconds"]) != 90:
        raise ValueError("settlement_tolerance_window_seconds deve ser 90")
    if int(inv["post_settlement_is_trading_wait_seconds"]) != 90:
        raise ValueError("post_settlement_is_trading_wait_seconds deve ser 90")
    if int(inv["amort_cycles_min"]) != 2 or int(inv["amort_cycles_max"]) != 4:
        raise ValueError("amort_cycles deve ser 2-4")
    if abs(float(inv["cover_multiple"]) - 1.5) > 1e-9:
        raise ValueError("cover_multiple deve ser 1.5")
    if abs(float(inv["max_safe_stake_pct_linear3"]) - 0.025) > 1e-9:
        raise ValueError("max_safe_stake_pct_linear3 deve ser 0.025")
    if abs(float(inv["large_account_stop_win_pct"]) - 3.0) > 1e-9:
        raise ValueError("large_account_stop_win_pct deve ser 3.0")
    if float(inv["min_validation_accuracy_gate"]) + 1e-12 < _PRODUCTION_MIN_ACC:
        raise ValueError(f"min_validation_accuracy_gate < {_PRODUCTION_MIN_ACC}")
    if float(inv["explore_stake_scale_floor"]) <= 0.0:
        raise ValueError("explore_stake_scale_floor deve ser > 0")
    if float(inv["max_safe_stake_cap"]) <= 0.0 or float(inv["max_safe_stake_pct"]) <= 0.0:
        raise ValueError("max_safe_stake_cap/pct devem ser > 0")
    if inv["signal_skip_enabled"]:
        floor = float(inv["signal_skip_min_direction_margin"])
        if floor + 1e-12 < 0.015 or floor - 1e-12 > 0.05:
            raise ValueError("signal_skip.min_direction_margin deve estar em [0.015, 0.05]")
    return inv
