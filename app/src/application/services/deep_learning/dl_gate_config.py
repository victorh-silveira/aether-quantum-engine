"""Configuracao do gate de deploy Deep Learning sem dependencias de inferencia."""

from typing import Any


def parse_deploy_gate_config(dl_config: dict) -> dict[str, Any]:
    """Extrai configuracao do bloco deploy_gate."""
    raw = dl_config.get("deploy_gate", {}) if isinstance(dl_config.get("deploy_gate"), dict) else {}
    return {
        "enabled": bool(raw.get("enabled", True)),
        "max_brier": float(raw.get("max_brier", 0.24)),
        "min_win_rate": float(raw.get("min_win_rate", 0.52)),
        "mini_bars": int(raw.get("mini_bars", 80)),
        "max_eval_steps": int(raw.get("max_eval_steps", 160)),
        "min_trades": int(raw.get("min_trades", 8)),
        "soft_min_val_accuracy": float(raw.get("soft_min_val_accuracy", 0.50)),
        "soft_max_brier": float(raw.get("soft_max_brier", 0.32)),
        "eval_relaxed_gating": bool(raw.get("eval_relaxed_gating", True)),
    }


def deploy_params_for_eval(params: dict[str, Any], gate_cfg: dict[str, Any]) -> dict[str, Any]:
    """Parametros mais permissivos para mini walk-forward de deploy."""
    if not gate_cfg.get("eval_relaxed_gating", True):
        return params
    out = dict(params)
    out["min_conviction"] = min(float(params.get("min_conviction", 0.58)), 0.52)
    out["min_edge_margin"] = min(float(params.get("min_edge_margin", 0.06)), 0.04)
    out["min_val_accuracy"] = 0.0
    out["max_val_brier_execute"] = max(float(params.get("max_val_brier_execute", 0.28)), 0.35)
    return out


def resolve_deploy_ok(
    *,
    mini_ok: bool,
    val_accuracy: float,
    val_brier: float,
    gate_cfg: dict[str, Any],
) -> bool:
    """Combina mini deploy com fallback por metricas de treino."""
    if mini_ok:
        return True
    if not gate_cfg.get("enabled", True):
        return True
    soft_acc = float(gate_cfg.get("soft_min_val_accuracy", 0.50))
    soft_brier = float(gate_cfg.get("soft_max_brier", 0.32))
    return float(val_accuracy) + 1e-9 >= soft_acc and float(val_brier) + 1e-9 <= soft_brier
