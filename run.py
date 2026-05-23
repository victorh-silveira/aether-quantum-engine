"""Atalho na raiz do repositorio para app/run.py."""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parent
    target = repo / "app" / "run.py"
    raise SystemExit(subprocess.call([sys.executable, str(target)], cwd=repo))


if __name__ == "__main__":
    main()
