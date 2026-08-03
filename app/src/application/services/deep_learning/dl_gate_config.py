"""Configuracao parametrica de deploy gate a partir de settings."""

from typing import Any

from src.domain.config_knobs import merge_settings_block, require_bool, require_float, require_int, require_keys


_DEPLOY_GATE_KEYS = (
    "enabled",
    "force_ok",
    "max_brier",
    "min_win_rate",
    "mini_bars",
    "max_eval_steps",
    "min_trades",
    "soft_min_val_accuracy",
    "soft_max_brier",
    "eval_relaxed_gating",
    "eval_call_threshold_cap",
    "eval_put_threshold_floor",
    "eval_call_threshold_default",
    "eval_put_threshold_default",
)


def parse_deploy_gate_config(dl_config: dict) -> dict[str, Any]:
    """Resolve deploy_gate com merge de override parcial sobre o SSOT."""
    override = dl_config.get("deploy_gate") if isinstance(dl_config, dict) else None
    raw = merge_settings_block(
        ("deep_learning", "deploy_gate"),
        override if isinstance(override, dict) else None,
    )
    block = require_keys(raw, _DEPLOY_GATE_KEYS, "deep_learning.deploy_gate")
    return {
        "enabled": require_bool(block, "enabled"),
        "force_ok": require_bool(block, "force_ok"),
        "max_brier": require_float(block, "max_brier"),
        "min_win_rate": require_float(block, "min_win_rate"),
        "mini_bars": require_int(block, "mini_bars"),
        "max_eval_steps": require_int(block, "max_eval_steps"),
        "min_trades": require_int(block, "min_trades"),
        "soft_min_val_accuracy": require_float(block, "soft_min_val_accuracy"),
        "soft_max_brier": require_float(block, "soft_max_brier"),
        "eval_relaxed_gating": require_bool(block, "eval_relaxed_gating"),
        "eval_call_threshold_cap": require_float(block, "eval_call_threshold_cap"),
        "eval_put_threshold_floor": require_float(block, "eval_put_threshold_floor"),
        "eval_call_threshold_default": require_float(block, "eval_call_threshold_default"),
        "eval_put_threshold_default": require_float(block, "eval_put_threshold_default"),
    }


def deploy_params_for_eval(params: dict[str, Any], gate_cfg: dict[str, Any]) -> dict[str, Any]:
    """Parametros mais permissivos para mini walk-forward de deploy."""
    if not bool(gate_cfg.get("eval_relaxed_gating", True)):
        return params
    out = dict(params)
    out["min_val_accuracy"] = 0.0
    call_cap = float(gate_cfg["eval_call_threshold_cap"]) if "eval_call_threshold_cap" in gate_cfg else 0.65
    put_floor = float(gate_cfg["eval_put_threshold_floor"]) if "eval_put_threshold_floor" in gate_cfg else 0.35
    call_default = float(gate_cfg["eval_call_threshold_default"]) if "eval_call_threshold_default" in gate_cfg else 0.75
    put_default = float(gate_cfg["eval_put_threshold_default"]) if "eval_put_threshold_default" in gate_cfg else 0.25
    call_thr = float(out["confidence_call_threshold"]) if "confidence_call_threshold" in out else call_default
    put_thr = float(out["confidence_put_threshold"]) if "confidence_put_threshold" in out else put_default
    out["confidence_call_threshold"] = min(call_thr, call_cap)
    out["confidence_put_threshold"] = max(put_thr, put_floor)
    return out


def resolve_deploy_ok(
    *,
    mini_ok: bool,
    val_accuracy: float,
    val_brier: float,
    gate_cfg: dict[str, Any],
) -> bool:
    """Combina mini deploy com fallback por metricas de treino; ACC soft e piso duro."""
    soft_acc = float(gate_cfg.get("soft_min_val_accuracy", 0.53))
    soft_brier = float(gate_cfg.get("soft_max_brier", 0.32))
    if float(val_accuracy) + 1e-9 < soft_acc:
        return False
    if bool(gate_cfg.get("force_ok", False)):
        return True
    if mini_ok:
        return True
    if not bool(gate_cfg.get("enabled", True)):
        return float(val_brier) + 1e-9 <= soft_brier
    return float(val_brier) + 1e-9 <= soft_brier
