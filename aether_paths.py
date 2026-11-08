"""Raizes do monorepo: codigo unificado em src/ e raiz do repositorio."""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
APP_ROOT = REPO_ROOT
SRC_ROOT = REPO_ROOT / "src"

for p in (str(REPO_ROOT), str(SRC_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


def repo_path(*parts: str) -> Path:
    """Monta caminho absoluto sob a raiz do repositorio."""
    return REPO_ROOT.joinpath(*parts)
