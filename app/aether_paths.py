"""Raizes do monorepo: app/ (codigo) e repositorio (config, docs, dados)."""

import sys
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parent

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


def repo_path(*parts: str) -> Path:
    """Monta caminho absoluto sob a raiz do repositorio."""
    return REPO_ROOT.joinpath(*parts)
