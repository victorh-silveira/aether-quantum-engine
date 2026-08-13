"""Candidatos e knobs do sweep multi-TF (offline)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aether_paths import REPO_ROOT, repo_path
from src.application.services.deep_learning.tf_sweep_scale import (
    scale_history,
    scale_lookback,
)
from src.domain.config_knobs import load_settings_json, require_keys
from src.infrastructure.api.deriv_granularity import DERIV_ALLOWED_GRANULARITY_SECONDS


_CANDIDATE_KEYS = (
    "tf",
    "enabled",
    "micro_granularity",
    "macro_granularity",
    "duration",
)


def default_candidates_path() -> Path:
    """Path SSOT do manifesto de candidatos."""
    return repo_path("config", "tf_sweep_candidates.json")


def load_tf_sweep_knobs(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """Le deep_learning.tf_sweep com defaults fail-closed."""
    full = settings if isinstance(settings, dict) else load_settings_json()
    dl = full.get("deep_learning") if isinstance(full.get("deep_learning"), dict) else {}
    raw = dl.get("tf_sweep") if isinstance(dl, dict) else None
    block = raw if isinstance(raw, dict) else {}
    payout = _resolve_payout(full, block)
    return {
        "enabled": bool(block.get("enabled", True)),
        "run_in_launch_train": bool(block.get("run_in_launch_train", True)),
        "auto_promote": bool(block.get("auto_promote", True)),
        "train_deploy_retries": max(1, min(8, int(block.get("train_deploy_retries", 1)))),
        "disable_infra_during_sweep": bool(block.get("disable_infra_during_sweep", True)),
        "candidates_path": str(block.get("candidates_path", "config/tf_sweep_candidates.json")),
        "min_edge_vs_breakeven": float(block.get("min_edge_vs_breakeven", 0.03)),
        "min_settle_n": max(1, int(block.get("min_settle_n", 16))),
        "min_history_bars": max(0, int(block.get("min_history_bars", 800))),
        "payout_for_breakeven": float(payout),
        "weight_edge": float(block.get("weight_edge", 1.0)),
        "weight_brier": float(block.get("weight_brier", 0.5)),
        "weight_sharpness": float(block.get("weight_sharpness", 0.25)),
        "weight_meta_ir": float(block.get("weight_meta_ir", 0.25)),
        "leaderboard_path": str(block.get("leaderboard_path", "data/dl/sweep/leaderboard.json")),
        "artifact_root": str(block.get("artifact_root", "data/dl/sweep")),
        "soft_max_brier": float(block.get("soft_max_brier", 0.26)),
    }


def _resolve_payout(full: dict[str, Any], block: dict[str, Any]) -> float:
    """Resolve payout de breakeven do bloco sweep ou risk_management.params."""
    if block.get("payout_for_breakeven") is not None:
        return float(block["payout_for_breakeven"])
    risk = full.get("risk_management") if isinstance(full.get("risk_management"), dict) else {}
    params = risk.get("params") if isinstance(risk, dict) and isinstance(risk.get("params"), dict) else {}
    if isinstance(params, dict) and params.get("payout_estimate") is not None:
        return float(params["payout_estimate"])
    return 0.72


def load_tf_sweep_manifest(path: Path | None = None) -> dict[str, Any]:
    """Carrega manifesto JSON de candidatos."""
    target = path if path is not None else default_candidates_path()
    if not target.is_file():
        raise FileNotFoundError(f"manifesto TF sweep ausente: {target}")
    raw = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("tf_sweep_candidates.json invalido")
    require_keys(raw, ("version", "defaults", "candidates"), "tf_sweep_candidates")
    if not isinstance(raw["defaults"], dict):
        raise ValueError("tf_sweep_candidates.defaults obrigatorio")
    if not isinstance(raw["candidates"], list):
        raise ValueError("tf_sweep_candidates.candidates obrigatorio")
    return raw


def resolve_enabled_candidates(
    manifest: dict[str, Any] | None = None,
    *,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Normaliza candidatos enabled com defaults e gran Deriv."""
    data = manifest if isinstance(manifest, dict) else load_tf_sweep_manifest(path)
    defaults = data["defaults"] if isinstance(data.get("defaults"), dict) else {}
    out: list[dict[str, Any]] = []
    for item in data.get("candidates") or []:
        if not isinstance(item, dict):
            continue
        require_keys(item, _CANDIDATE_KEYS, "tf_sweep_candidates.candidate")
        if not bool(item["enabled"]):
            continue
        micro = int(item["micro_granularity"])
        macro = int(item["macro_granularity"])
        if micro not in DERIV_ALLOWED_GRANULARITY_SECONDS:
            raise ValueError(f"micro_granularity invalida Deriv: {micro}")
        if macro not in DERIV_ALLOWED_GRANULARITY_SECONDS:
            raise ValueError(f"macro_granularity invalida Deriv: {macro}")
        duration_unit = str(item.get("duration_unit") or defaults.get("duration_unit") or "m")
        lookback = int(item["lookback"]) if item.get("lookback") is not None else scale_lookback(micro)
        history = int(item["history_bars"]) if item.get("history_bars") is not None else scale_history(micro, lookback)
        row = {
            "tf": str(item["tf"]).strip().upper(),
            "enabled": True,
            "micro_granularity": micro,
            "macro_granularity": macro,
            "mini_granularity": int(item.get("mini_granularity") or micro),
            "duration": int(item["duration"]),
            "duration_unit": duration_unit,
            "lookback": lookback,
            "history_bars": history,
            "label_horizon_bars": int(item.get("label_horizon_bars") or defaults.get("label_horizon_bars") or 1),
            "train_timeframe": str(item.get("train_timeframe") or defaults.get("train_timeframe") or "micro"),
        }
        out.append(row)
    return out


def resolve_repo_path(rel: str, *, repo_root: Path | None = None) -> Path:
    """Resolve path relativo ao root do repositorio."""
    root = repo_root if repo_root is not None else REPO_ROOT
    path = Path(rel)
    return path if path.is_absolute() else root / path


def candidate_artifact_dir(artifact_root: str, tf: str, *, repo_root: Path | None = None) -> Path:
    """Diretorio isolado data/dl/sweep/{tf}."""
    return resolve_repo_path(artifact_root, repo_root=repo_root) / str(tf).strip().upper()


def candidate_model_template(artifact_root: str, tf: str) -> str:
    """Template de ckpt isolado por TF no sweep."""
    root = artifact_root.replace("\\", "/").rstrip("/")
    return f"{root}/{str(tf).strip().upper()}/{{symbol}}.pth"
