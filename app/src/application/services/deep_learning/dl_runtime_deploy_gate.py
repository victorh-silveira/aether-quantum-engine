"""Autorizacao de checkpoint identica para contas virtuais e reais."""

from src.application.services.deep_learning.dl_gate_config import parse_deploy_gate_config


def apply_deploy_gate(entry: dict, runtime: dict, dl_config: dict, orch=None) -> dict:
    """Separa validade tecnica do checkpoint da qualificacao estatistica OOS."""
    del orch
    gate_cfg = parse_deploy_gate_config(dl_config)
    metrics = entry["metrics"]
    checkpoint_ready = bool(runtime.get("checkpoint_loaded", False)) and bool(runtime.get("session_trained", False))
    qualified = checkpoint_ready and bool(runtime.get("deploy_ok", False))
    provisional = (
        checkpoint_ready
        and not qualified
        and bool(runtime.get("deploy_provisional_ok", False))
        and bool(gate_cfg["provisional_enabled"])
    )
    exploration = checkpoint_ready and not qualified and not provisional
    if not checkpoint_ready and metrics.get("execute"):
        metrics["execute"] = False
        metrics["gate_reason"] = "model_unavailable"
    metrics["deploy_ok"] = checkpoint_ready
    metrics["model_deploy_qualified"] = qualified
    metrics["checkpoint_exploration"] = exploration
    metrics["deploy_provisional"] = provisional or exploration
    if exploration:
        metrics["provisional_max_stake_pct"] = min(0.01, max(0.0, float(gate_cfg["unqualified_max_stake_pct"])))
    elif provisional:
        metrics["provisional_max_stake_pct"] = float(gate_cfg["provisional_max_stake_pct"])
    return entry
