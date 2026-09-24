"""Configuracao parametrica de deploy gate a partir de settings."""

from typing import Any

from src.domain.config_knobs import merge_settings_block, require_bool, require_float, require_int, require_keys


_DEPLOY_GATE_KEYS = (
    "enabled",
    "force_ok",
    "enforce_settle_gate",
    "max_brier",
    "min_win_rate",
    "provisional_enabled",
    "provisional_max_brier",
    "provisional_max_stake_pct",
    "provisional_min_trades",
    "provisional_min_win_rate",
    "settlement_confidence",
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
    "reject_majority_collapse",
    "require_broker_settlement",
    "max_label_call_frac_bias",
    "min_minority_recall",
    "unqualified_max_stake_pct",
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
        "enforce_settle_gate": require_bool(block, "enforce_settle_gate"),
        "max_brier": require_float(block, "max_brier"),
        "min_win_rate": require_float(block, "min_win_rate"),
        "provisional_enabled": require_bool(block, "provisional_enabled"),
        "provisional_max_brier": require_float(block, "provisional_max_brier"),
        "provisional_max_stake_pct": require_float(block, "provisional_max_stake_pct"),
        "provisional_min_trades": require_int(block, "provisional_min_trades"),
        "provisional_min_win_rate": require_float(block, "provisional_min_win_rate"),
        "settlement_confidence": require_float(block, "settlement_confidence"),
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
        "reject_majority_collapse": require_bool(block, "reject_majority_collapse"),
        "require_broker_settlement": require_bool(block, "require_broker_settlement"),
        "max_label_call_frac_bias": require_float(block, "max_label_call_frac_bias"),
        "min_minority_recall": require_float(block, "min_minority_recall"),
        "unqualified_max_stake_pct": require_float(block, "unqualified_max_stake_pct"),
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


def _pred_side_skew(
    *,
    label_call_frac: float | None,
    pred_call_frac: float | None,
    bias_cap: float,
) -> bool:
    """True quando a fracao predita foge de 0.5 ou do label alem do bias_cap."""
    if pred_call_frac is None:
        return False
    pred = float(pred_call_frac)
    if abs(pred - 0.5) > bias_cap:
        return True
    return label_call_frac is not None and abs(pred - float(label_call_frac)) > bias_cap


def _majority_collapse_hit(
    gate_cfg: dict[str, Any],
    *,
    label_call_frac: float | None,
    pred_call_frac: float | None,
    minority_recall: float | None,
) -> bool:
    """True em colapso preditivo ou label viesado com minority_recall abaixo do piso."""
    if not bool(gate_cfg.get("reject_majority_collapse", False)):
        return False
    bias_cap = float(gate_cfg.get("max_label_call_frac_bias", 0.20))
    min_rec = float(gate_cfg.get("min_minority_recall", 0.25))
    if _pred_side_skew(
        label_call_frac=label_call_frac,
        pred_call_frac=pred_call_frac,
        bias_cap=bias_cap,
    ):
        return True
    if minority_recall is None:
        return False
    if float(minority_recall) + 1e-9 >= min_rec:
        return False
    return label_call_frac is not None and abs(float(label_call_frac) - 0.5) > bias_cap


def resolve_deploy_ok(
    *,
    mini_ok: bool,
    val_accuracy: float,
    val_brier: float,
    gate_cfg: dict[str, Any],
    label_call_frac: float | None = None,
    pred_call_frac: float | None = None,
    minority_recall: float | None = None,
) -> bool:
    """Aceita somente evidencia OOS de settlement, sem colapso preditivo."""
    soft_acc = float(gate_cfg.get("soft_min_val_accuracy", 0.53))
    soft_brier = float(gate_cfg.get("soft_max_brier", 0.26))
    if float(val_accuracy) + 1e-9 < soft_acc:
        return False
    if _majority_collapse_hit(
        gate_cfg,
        label_call_frac=label_call_frac,
        pred_call_frac=pred_call_frac,
        minority_recall=minority_recall,
    ):
        return False
    _ = (val_brier, soft_brier)
    if not bool(gate_cfg.get("enforce_settle_gate", True)):
        return True
    return bool(mini_ok) or not bool(gate_cfg.get("enabled", True))


def resolve_provisional_deploy_ok(
    *,
    provisional_ok: bool,
    val_accuracy: float,
    gate_cfg: dict[str, Any],
    label_call_frac: float | None = None,
    pred_call_frac: float | None = None,
    minority_recall: float | None = None,
) -> bool:
    """Aceita modo provisório somente com sinal OOS e anti-colapso intactos."""
    if not bool(gate_cfg.get("provisional_enabled", False)):
        return False
    if float(val_accuracy) + 1e-9 < float(gate_cfg.get("soft_min_val_accuracy", 0.50)):
        return False
    if _majority_collapse_hit(
        gate_cfg,
        label_call_frac=label_call_frac,
        pred_call_frac=pred_call_frac,
        minority_recall=minority_recall,
    ):
        return False
    if not bool(gate_cfg.get("enforce_settle_gate", True)):
        return True
    return bool(provisional_ok)


def describe_deploy_block(
    *,
    mini_ok: bool,
    val_accuracy: float,
    val_brier: float,
    gate_cfg: dict[str, Any],
    label_call_frac: float | None = None,
    pred_call_frac: float | None = None,
    minority_recall: float | None = None,
) -> str:
    """Mensagem acionavel quando deploy_ok=false."""
    soft_acc = float(gate_cfg.get("soft_min_val_accuracy", 0.53))
    soft_brier = float(gate_cfg.get("soft_max_brier", 0.26))
    if float(val_accuracy) + 1e-9 < soft_acc:
        return f"ACC={val_accuracy:.4f}<soft_min={soft_acc:.4f}"
    if _majority_collapse_hit(
        gate_cfg,
        label_call_frac=label_call_frac,
        pred_call_frac=pred_call_frac,
        minority_recall=minority_recall,
    ):
        pred_s = f"{float(pred_call_frac):.3f}" if pred_call_frac is not None else "n/a"
        label_s = f"{float(label_call_frac):.3f}" if label_call_frac is not None else "n/a"
        rec_s = f"{float(minority_recall):.3f}" if minority_recall is not None else "n/a"
        return f"majority_collapse label_call={label_s} pred_call={pred_s} minority_rec={rec_s}"
    if mini_ok:
        return "mini_ok mas gate rejeitou (inesperado)"
    return f"settlement OOS nao qualificou (val_brier={val_brier:.4f}; soft_max_brier={soft_brier:.4f})"
