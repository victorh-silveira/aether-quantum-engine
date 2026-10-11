"""Atalho na raiz do repositorio para app/run.py."""

import os
import sys
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parent
    target = repo / "app" / "run.py"
    os.chdir(repo)
    os.execv(sys.executable, [sys.executable, str(target)])


if __name__ == "__main__":
    main()
