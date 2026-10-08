"""Autorizacao de checkpoint identica para contas virtuais e reais."""

from src.domain.risk.checkpoint_stake_cap import MAX_CHECKPOINT_STAKE_PCT


def apply_deploy_gate(entry: dict, runtime: dict, dl_config: dict, orch=None) -> dict:
    """Exige checkpoint carregado e limita a stake a 1% da banca."""
    del orch
    metrics = entry["metrics"]
    checkpoint_ready = bool(runtime.get("checkpoint_loaded", False)) and bool(runtime.get("session_trained", False))
    exploration = checkpoint_ready
    if not checkpoint_ready and metrics.get("execute"):
        metrics["execute"] = False
        metrics["gate_reason"] = "model_unavailable"
    metrics["deploy_ok"] = checkpoint_ready
    metrics["model_deploy_qualified"] = False
    metrics["checkpoint_exploration"] = exploration
    metrics["deploy_provisional"] = exploration
    if exploration:
        metrics["provisional_max_stake_pct"] = min(
            MAX_CHECKPOINT_STAKE_PCT,
            max(0.0, float(dl_config.get("checkpoint_max_stake_pct", MAX_CHECKPOINT_STAKE_PCT))),
        )
    return entry
