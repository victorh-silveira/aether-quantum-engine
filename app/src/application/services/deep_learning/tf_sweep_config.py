"""Knobs e paths do sweep de horizonte (celulas H1-H5)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aether_paths import REPO_ROOT
from src.application.services.deep_learning.horizon_sweep import load_horizon_sweep_knobs


def load_tf_sweep_knobs(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """Alias dos knobs de horizon_sweep (pisos settle / artefactos)."""
    return load_horizon_sweep_knobs(settings)


def resolve_repo_path(rel: str, *, repo_root: Path | None = None) -> Path:
    """Resolve path relativo ao root do repositorio."""
    root = repo_root if repo_root is not None else REPO_ROOT
    path = Path(rel)
    return path if path.is_absolute() else root / path


def candidate_artifact_dir(
    artifact_root: str,
    tf: str,
    *,
    symbol: str = "1HZ75V",
    repo_root: Path | None = None,
) -> Path:
    """Diretorio isolado data/dl/sweep/{symbol}/{tf}."""
    sym = str(symbol).strip().upper()
    return resolve_repo_path(artifact_root, repo_root=repo_root) / sym / str(tf).strip().upper()


def candidate_model_template(artifact_root: str, tf: str, *, symbol: str = "1HZ75V") -> str:
    """Template de ckpt isolado por simbolo+celula no sweep."""
    root = artifact_root.replace("\\", "/").rstrip("/")
    sym = str(symbol).strip().upper()
    return f"{root}/{sym}/{str(tf).strip().upper()}/{{symbol}}.pth"
