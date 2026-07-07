"""Rotinas de limpeza de workspace e artefatos de runtime."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from scripts.operations.clean_runtime_artifacts import (
    clean_runtime_artifacts,
    docker_bind_mount_roots,
    is_docker_bind_mount,
)


_CACHE_NAMES = (
    ".pytest_cache",
    ".ruff_cache",
    ".coverage",
    "htmlcov",
    "dist",
    "build",
    ".mypy_cache",
)
_SKIP_WALK_DIRS = frozenset({".venv", "venv", ".git", ".idea", ".vscode"})


def build_safe_remove(repo_root: Path):
    """Constroi removedor seguro que ignora bind mounts Docker."""
    preserved = docker_bind_mount_roots(repo_root)

    def safe_remove(path: Path) -> None:
        if is_docker_bind_mount(path, repo_root):
            print(f"Preservado bind mount Docker: {path}")
            return
        try:
            if path.is_dir():
                shutil.rmtree(path)
                print(f"Removido diretório: {path}")
            else:
                path.unlink()
                print(f"Removido arquivo: {path}")
        except Exception as exc:
            print(f"Erro ao remover {path}: {exc}")

    return safe_remove, preserved


def clean_named_caches(scan_root: Path, safe_remove) -> None:
    """Remove caches nomeados de ferramentas de desenvolvimento."""
    for name in _CACHE_NAMES:
        candidate = scan_root / name
        if candidate.exists():
            safe_remove(candidate)


def clean_python_artifacts(scan_root: Path, safe_remove, repo_root: Path) -> None:
    """Remove bytecode Python sem entrar em bind mounts Docker."""
    for root, dirs, files in os.walk(scan_root):
        dirs[:] = [
            entry
            for entry in dirs
            if entry not in _SKIP_WALK_DIRS and not is_docker_bind_mount(Path(root) / entry, repo_root)
        ]
        for entry in list(dirs):
            if entry == "__pycache__":
                safe_remove(Path(root) / entry)
                dirs.remove(entry)
        for filename in files:
            if filename.endswith((".pyc", ".pyo", ".pyd")):
                safe_remove(Path(root) / filename)


def clean_workspace_artifacts(app_root: Path, repo_root: Path, safe_remove) -> None:
    """Limpa caches, logs e dados efemeros do workspace."""
    for scan_root in (app_root, repo_root):
        clean_named_caches(scan_root, safe_remove)
        clean_python_artifacts(scan_root, safe_remove, repo_root)
    logs_dir = repo_root / "logs"
    if logs_dir.exists():
        safe_remove(logs_dir)
    app_data = app_root / "data"
    if app_data.exists():
        safe_remove(app_data)
    for pattern in (repo_root.glob("pytest-cache-files-*"), app_root.glob("pytest-cache-files-*")):
        for stray in pattern:
            safe_remove(stray)


def stage_clean(app_root: Path, repo_root: Path) -> None:
    """Executa limpeza completa preservando bind mounts montados nos containers."""
    print("\n>>> Running: Limpeza de lixo e caches")
    safe_remove, preserved_mounts = build_safe_remove(repo_root)
    for mount in preserved_mounts:
        print(f"Bind mount Docker preservado: {mount}")
    clean_workspace_artifacts(app_root, repo_root, safe_remove)
    print("\n>>> Running: Limpeza de dados locais de run/treino (bind mounts Docker preservados)")
    clean_runtime_artifacts(repo_root, safe_remove)
